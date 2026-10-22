"""Places tracks on one continuous output timeline and renders it.

Every track is a Placement: an output-time window, a RateCurve mapping output time to source position,
and volume and low-band envelopes for how it enters and leaves. Transitions are planned by
`Timeline.append`, which trims the previous placement and positions the next one so their beat grids
coincide. Work too slow to do while rendering (master-tempo pitch correction, the low/high band split
for bass swaps) is done ahead of time by `Placement.prepare`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from autoplayer.audio.decode import SAMPLE_RATE, decode
from autoplayer.audio.eq import split_lows
from autoplayer.audio.keylock import restore_pitch
from autoplayer.audio.onsets import kick_offset, onset_lag
from autoplayer.audio.varispeed import RateCurve, sample_cubic
from autoplayer.library.anlz import Analysis, Beat, load_analysis
from autoplayer.library.pdb import Track

MixMode = Literal["off", "timed", "beatmatch"]

SILENCE_THRESHOLD = 10 ** (-50 / 20)
TEMPO_FACTORS = (1.0, 0.5, 2.0)
AUTO_CUE_COMMENT = "CUE(Auto)"
KEY_LOCK_MIN_TEMPO_CHANGE = 0.002
KEY_LOCK_SPLICE_FRAMES = 2048
KEY_LOCK_CHUNK_FRAMES = 30 * SAMPLE_RATE
KEY_LOCK_PREROLL_FRAMES = SAMPLE_RATE // 2
KEY_LOCK_OVERLAP_FRAMES = SAMPLE_RATE // 4
EQ_PAD_FRAMES = SAMPLE_RATE // 2
LOW_SWAP_BEATS = 0.25
CLICK_GUARD_FRAMES = int(0.005 * SAMPLE_RATE)
EQUAL_POWER_STEPS = 32


@dataclass(frozen=True)
class MixSettings:
    mode: MixMode = "beatmatch"
    overlap_bars: int = 16
    ease_bars: int = 8
    max_tempo_change: float = 0.08
    fade_seconds: float = 8.0
    key_lock: bool = True
    incoming_start_level: float = 0.5
    outgoing_end_level: float = 0.8
    hold_tempo: bool = False


@dataclass
class LoadedTrack:
    index: int
    title: str
    bpm: float
    audio: np.ndarray
    analysis: Analysis
    kick_offset_s: float = 0.0


def load_track(track: Track, index: int) -> LoadedTrack:
    audio = decode(track.audio_path)
    analysis = load_analysis(track.anlz_path)
    return LoadedTrack(
        index=index,
        title=track.title,
        bpm=track.bpm,
        audio=audio,
        analysis=analysis,
        kick_offset_s=kick_offset(audio, analysis.beats),
    )


@dataclass(frozen=True)
class Envelope:
    """A level over output time: linear between points and held flat beyond them."""

    times: tuple[float, ...]
    values: tuple[float, ...]

    @classmethod
    def equal_power(cls, start: float, length: float, rising: bool) -> Envelope:
        x = np.linspace(0, 1, EQUAL_POWER_STEPS + 1)
        values = np.sin(x * np.pi / 2) if rising else np.cos(x * np.pi / 2)
        return cls(tuple(start + x * length), tuple(values))

    @property
    def start(self) -> float:
        return self.times[0]

    @property
    def end(self) -> float:
        return self.times[-1]

    def at(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(t, self.times, self.values)


def _level(envelopes: tuple[Envelope | None, ...], t: np.ndarray) -> np.ndarray:
    level = np.ones_like(t, dtype=np.float64)
    for envelope in envelopes:
        if envelope is not None:
            level *= envelope.at(t)
    return level


@dataclass(frozen=True)
class Segment:
    start: int
    audio: np.ndarray

    @property
    def end(self) -> int:
        return self.start + self.audio.shape[0]


@dataclass(eq=False)
class Placement:
    item: LoadedTrack
    curve: RateCurve
    out_start: int
    out_end: int
    fade_in: Envelope | None = None
    fade_out: Envelope | None = None
    lows_in: Envelope | None = None
    lows_out: Envelope | None = None
    switch_at: float | None = None
    overlap_in_end: float | None = None
    beatmatched: bool = False
    cues: str = ""
    key_lock_end: int | None = None
    key_locked_audio: np.ndarray | None = None
    key_locked_until: int = 0
    eq_regions: list[tuple[int, int]] = field(default_factory=list)
    eq_segments: list[Segment] = field(default_factory=list)
    prev_before: dict | None = None
    cancelled: bool = False
    align_with: Placement | None = None

    @property
    def has_pending_work(self) -> bool:
        return self.align_with is not None or self._awaiting_key_lock or bool(self.eq_regions)

    @property
    def _key_lock_stop(self) -> int:
        """Where key-locked audio is needed until: the splice back to the original, or the track's end."""
        return min(self.key_lock_end + KEY_LOCK_SPLICE_FRAMES, self.out_end) if self.key_lock_end is not None else 0

    @property
    def _awaiting_key_lock(self) -> bool:
        return self.key_lock_end is not None and max(self.key_locked_until, self.out_start) < self._key_lock_stop

    def pending_between(self, start: int, end: int) -> bool:
        """Whether unprepared work overlaps mix samples [start, end)."""
        if self.align_with is not None and start < self.out_end and end > self.out_start:
            return True
        ready = max(self.key_locked_until, self.out_start)
        if self._awaiting_key_lock and start < self._key_lock_stop and end > ready:
            return True
        return any(start < region_end and end > region_start for region_start, region_end in self.eq_regions)

    def prepare(self) -> None:
        """Do the slow rendering ahead of time. Key lock comes in chunks, enough first for the band split
        (which works on its output), so a long held track is playable long before it's all corrected."""
        if self.align_with is not None:
            self._align()
        eq_needs = max((end for _, end in self.eq_regions), default=self.out_start)
        while self._awaiting_key_lock and self.key_locked_until < eq_needs and not self.cancelled:
            self._prepare_key_lock_chunk()
        for region in list(self.eq_regions):
            segment = self._render_eq(*region)
            if region in self.eq_regions:
                self.eq_segments = [*self.eq_segments, segment]
                self.eq_regions = [r for r in self.eq_regions if r != region]
        while self._awaiting_key_lock and not self.cancelled:
            self._prepare_key_lock_chunk()

    def _align(self) -> None:
        """Shift this track so its hits land with the outgoing track's in the overlap. Grids and kick shapes
        differ between tracks, so the attacks the ear hears are measured directly."""
        prev = self.align_with
        overlap = np.arange(self.out_start, int(prev.out_end), dtype=np.float64)
        lag = onset_lag(
            sample_cubic(prev.item.audio, prev.curve.src_position(overlap)),
            sample_cubic(self.item.audio, self.curve.src_position(overlap)),
        )
        if lag:
            curve = self.curve
            anchor = float(self.out_start)
            self.curve = RateCurve(
                list(zip(curve.xs, curve.rates)),
                anchor_out=anchor - lag * SAMPLE_RATE,
                anchor_src=float(curve.src_position(anchor)),
            )
            self.out_end = int(np.ceil(self.out_time_for_src(self.item.audio.shape[0])))
        self.align_with = None

    def _prepare_key_lock_chunk(self) -> None:
        stop = self._key_lock_stop
        if self.key_locked_audio is None or self.key_locked_audio.shape[0] < stop - self.out_start:
            grown = np.zeros((stop - self.out_start, 2), dtype=np.float32)
            if self.key_locked_audio is not None:
                grown[: self.key_locked_audio.shape[0]] = self.key_locked_audio
            self.key_locked_audio = grown
        start = max(self.key_locked_until, self.out_start)
        end = min(start + KEY_LOCK_CHUNK_FRAMES, stop)
        overlap = min(KEY_LOCK_OVERLAP_FRAMES, start - self.out_start)
        render_from = start - overlap - KEY_LOCK_PREROLL_FRAMES
        t = np.arange(render_from, end, dtype=np.float64)
        varispeed = sample_cubic(self.item.audio, self.curve.src_position(t))
        corrected = restore_pitch(varispeed, self.curve.rate_at(t))[KEY_LOCK_PREROLL_FRAMES:]
        a = start - overlap - self.out_start
        if overlap:
            fade_in = np.linspace(0, 1, overlap, dtype=np.float32)[:, None]
            self.key_locked_audio[a : a + overlap] = (
                self.key_locked_audio[a : a + overlap] * (1 - fade_in) + corrected[:overlap] * fade_in
            )
        self.key_locked_audio[a + overlap : end - self.out_start] = corrected[overlap:]
        self.key_locked_until = end

    def gain(self, t: np.ndarray) -> np.ndarray:
        return _level((self.fade_in, self.fade_out), t)

    def low_level(self, t: np.ndarray) -> np.ndarray:
        return _level((self.lows_in, self.lows_out), t)

    def src_seconds_at(self, t: float) -> float:
        return float(self.curve.src_position(t)) / SAMPLE_RATE

    def out_time_for_src(self, src: float) -> float:
        start_src = float(self.curve.src_position(self.out_start))
        span = abs(src - start_src) / min(float(self.curve.rates.min()), 1.0) + SAMPLE_RATE
        return self.curve.out_time_for_src(src, self.out_start - span, self.out_start + span)

    def render_into(self, mix: np.ndarray, t0: int) -> None:
        lo = max(t0, self.out_start)
        hi = min(t0 + mix.shape[0], self.out_end)
        if hi <= lo:
            return
        samples = self._deck(lo, hi)
        for segment in self.eq_segments:
            a, b = max(lo, segment.start), min(hi, segment.end)
            if a < b:
                samples[a - lo : b - lo] = segment.audio[a - segment.start : b - segment.start]
        t = np.arange(lo, hi, dtype=np.float64)
        mix[lo - t0 : hi - t0] += samples * self.gain(t)[:, None].astype(np.float32)

    def _deck(self, lo: int, hi: int) -> np.ndarray:
        """The track as it plays over output samples [lo, hi): tempo-mapped, key-locked where prepared."""
        t = np.arange(lo, hi, dtype=np.float64)
        samples = sample_cubic(self.item.audio, self.curve.src_position(t))
        locked = self.key_locked_audio
        ready = self.key_locked_until - self.out_start
        if locked is not None and lo < self.out_start + ready and hi > self.out_start:
            offsets = np.arange(lo, hi) - self.out_start
            inside = (offsets >= 0) & (offsets < ready)
            weight = np.clip((self.key_lock_end + KEY_LOCK_SPLICE_FRAMES - t[inside]) / KEY_LOCK_SPLICE_FRAMES, 0, 1)
            weight = weight[:, None].astype(np.float32)
            samples[inside] = locked[offsets[inside]] * weight + samples[inside] * (1 - weight)
        return samples

    def _render_eq(self, start: int, end: int) -> Segment:
        deck = self._deck(start - EQ_PAD_FRAMES, end + EQ_PAD_FRAMES)
        lows, rest = split_lows(deck)
        t = np.arange(start - EQ_PAD_FRAMES, end + EQ_PAD_FRAMES, dtype=np.float64)
        shaped = rest + lows * self.low_level(t)[:, None].astype(np.float32)
        return Segment(start, shaped[EQ_PAD_FRAMES : EQ_PAD_FRAMES + end - start])


_TRANSITION_OUT_FIELDS = ("out_end", "fade_out", "lows_out", "eq_regions", "eq_segments")


def audible_bounds(audio: np.ndarray) -> tuple[int, int]:
    loud = np.flatnonzero(np.max(np.abs(audio), axis=1) > SILENCE_THRESHOLD) if audio.size else np.array([])
    if loud.size == 0:
        return 0, audio.shape[0]
    return int(loud[0]), int(loud[-1]) + 1


def _beat_src(beats: tuple[Beat, ...]) -> np.ndarray:
    return np.array([b.time_s * SAMPLE_RATE for b in beats], dtype=np.float64)


def _downbeats(beats: tuple[Beat, ...]) -> list[int]:
    return [i for i, b in enumerate(beats) if b.bar_pos == 1]


def hot_cue_beats(analysis: Analysis) -> dict[str, int]:
    """Hand-placed hot cues by letter (A = 1), each snapped to its nearest beat.

    rekordbox's auto-generated hot cues are skipped: they repeat in groups and stack several letters at
    the start of the track, so they say nothing about where the DJ wants to mix.
    """
    if not analysis.beats:
        return {}
    times = np.array([b.time_s for b in analysis.beats])
    return {
        chr(ord("A") + cue.hot_cue - 1): int(np.argmin(np.abs(times - cue.time_s)))
        for cue in analysis.cues
        if cue.hot_cue > 0 and not cue.is_loop and cue.comment != AUTO_CUE_COMMENT
    }


def _ordered_cues(cues: dict[str, int], letters: str) -> dict[str, int]:
    """The cues among `letters`, or none of them unless they fall in strictly increasing order."""
    present = {letter: cues[letter] for letter in letters if letter in cues}
    beats = list(present.values())
    return present if all(x < y for x, y in zip(beats, beats[1:])) else {}


def _cue_label(out_cues: str, in_cues: str) -> str:
    """e.g. "EFG → ABC", or "ABC in" when only the incoming track is cued."""
    if out_cues and in_cues:
        return f"{out_cues} → {in_cues}"
    if out_cues:
        return f"{out_cues} out"
    return f"{in_cues} in" if in_cues else ""


@dataclass(frozen=True)
class MixPoints:
    """Beat indices for one transition: where the outgoing track starts its transition and how many beats
    it lasts, and where the incoming one enters. `out_mid` (F) or `in_mid` (B) places the bass swap;
    `in_end` (C) sizes the transition when the outgoing track has no G."""

    out_start: int
    out_beats: int
    in_start: int
    out_mid: int | None = None
    in_mid: int | None = None
    in_end: int | None = None
    cues: str = ""


def mix_in_beat(analysis: Analysis) -> int | None:
    """Index of the beat B should enter on: the first memory cue snapped to a downbeat, else the first downbeat."""
    downbeats = _downbeats(analysis.beats)
    if not downbeats:
        return None
    cues = analysis.memory_cues
    if not cues:
        return downbeats[0]
    times = np.array([analysis.beats[i].time_s for i in downbeats])
    return downbeats[int(np.argmin(np.abs(times - cues[0].time_s)))]


class Timeline:
    def __init__(self, settings: MixSettings):
        self.settings = settings
        self.placements: list[Placement] = []

    @property
    def end(self) -> int:
        return self.placements[-1].out_end if self.placements else 0

    def append(self, item: LoadedTrack, not_before: int = 0, start_src: float | None = None) -> Placement:
        if not self.placements:
            placement = self._place_first(item, not_before, start_src)
        else:
            prev = self.placements[-1]
            before = {name: getattr(prev, name) for name in _TRANSITION_OUT_FIELDS}
            placement = None
            if self.settings.mode == "beatmatch":
                placement = self._place_beatmatched(prev, item, not_before)
            if placement is None and self.settings.mode != "off":
                placement = self._place_timed(prev, item, not_before)
            if placement is None:
                placement = self._place_gapless(prev, item, not_before)
            placement.prev_before = before
        self.placements.append(placement)
        return placement

    def unplan_after(self, tau: float) -> int | None:
        """Drop placements that haven't started by mix time `tau`, restoring the track before each to how it was
        before that transition was planned. Returns the playlist index of the earliest one dropped."""
        first = None
        while len(self.placements) > 1 and self.placements[-1].out_start > tau:
            dropped = self.placements.pop()
            dropped.cancelled = True
            for name, value in dropped.prev_before.items():
                setattr(self.placements[-1], name, value)
            first = dropped.item.index
        return first

    def prepare(self) -> None:
        for placement in self.placements:
            if placement.has_pending_work:
                placement.prepare()

    def render(self, t0: int, n: int) -> np.ndarray:
        mix = np.zeros((n, 2), dtype=np.float32)
        for placement in list(self.placements):
            placement.render_into(mix, t0)
        return mix

    def _place_first(self, item: LoadedTrack, out_start: int, start_src: float | None) -> Placement:
        src = audible_bounds(item.audio)[0] if start_src is None else start_src
        curve = RateCurve([(out_start, 1.0)], anchor_out=out_start, anchor_src=src)
        return Placement(item, curve, out_start, out_start + int(item.audio.shape[0] - src))

    def _earliest_mix_start(self, prev: Placement, not_before: int) -> float:
        earliest = max(float(not_before), float(prev.out_start))
        if prev.overlap_in_end is not None:
            earliest = max(earliest, prev.overlap_in_end)
        if len(prev.curve.xs) > 1:
            earliest = max(earliest, float(prev.curve.xs[-1]))
        return earliest

    def _place_gapless(self, prev: Placement, item: LoadedTrack, not_before: int) -> Placement:
        a_end = prev.out_time_for_src(audible_bounds(prev.item.audio)[1])
        start = int(round(max(a_end, not_before)))
        prev.out_end = min(prev.out_end, start)
        b_start = audible_bounds(item.audio)[0]
        curve = RateCurve([(start, 1.0)], anchor_out=start, anchor_src=b_start)
        return Placement(item, curve, start, start + int(item.audio.shape[0] - b_start))

    def _intro_start(self, item: LoadedTrack) -> float:
        cue = _ordered_cues(hot_cue_beats(item.analysis), "ABC").get("A")
        if cue is not None:
            return item.analysis.beats[cue].time_s * SAMPLE_RATE
        return audible_bounds(item.audio)[0]

    def _place_timed(self, prev: Placement, item: LoadedTrack, not_before: int) -> Placement | None:
        fade_len = self.settings.fade_seconds * SAMPLE_RATE
        a_end = prev.out_time_for_src(audible_bounds(prev.item.audio)[1])
        start = a_end - fade_len
        if start < self._earliest_mix_start(prev, not_before):
            return None
        b_start = self._intro_start(item)
        prev.out_end = int(np.ceil(start + fade_len))
        prev.fade_out = Envelope.equal_power(start, fade_len, rising=False)
        curve = RateCurve([(start, 1.0)], anchor_out=start, anchor_src=b_start)
        out_start = int(np.floor(start))
        return Placement(
            item,
            curve,
            out_start,
            out_start + int(item.audio.shape[0] - b_start) + 1,
            fade_in=Envelope.equal_power(start, fade_len, rising=True),
            switch_at=start + fade_len / 2,
            overlap_in_end=start + fade_len,
        )

    def _place_beatmatched(self, prev: Placement, item: LoadedTrack, not_before: int) -> Placement | None:
        a_beats, b_beats = prev.item.analysis.beats, item.analysis.beats
        if len(a_beats) < 8 or not b_beats:
            return None
        a_src, b_src = _beat_src(a_beats), _beat_src(b_beats)
        earliest = self._earliest_mix_start(prev, not_before)
        for points in self._mix_point_candidates(prev, item):
            k, n_a, j = points.out_start, points.out_beats, points.in_start
            if k + n_a >= len(a_beats) or a_src[k + n_a] > prev.item.audio.shape[0]:
                continue
            t0 = prev.out_time_for_src(a_src[k])
            t1 = prev.out_time_for_src(a_src[k + n_a])
            if t0 < earliest:
                continue
            rates = [
                (b_src[j + int(n_a * factor)] - b_src[j]) / (t1 - t0)
                for factor in TEMPO_FACTORS
                if j + int(n_a * factor) < len(b_beats)
            ]
            if self.settings.hold_tempo:
                # A master BPM is set: every track plays at it, however far that is from its own tempo.
                rates = sorted(rates, key=lambda r: abs(np.log(r)))[:1]
            else:
                rates = [r for r in rates if abs(r - 1) <= self.settings.max_tempo_change][:1]
            for rate in rates:
                a_kick = prev.item.kick_offset_s * SAMPLE_RATE / float(prev.curve.rate_at(t0))
                b_kick = item.kick_offset_s * SAMPLE_RATE / rate
                return self._build_beatmatched(prev, item, points, t0, t1, t0 + a_kick - b_kick, rate)
        return None

    def _mix_point_candidates(self, prev: Placement, item: LoadedTrack) -> list[MixPoints]:
        """Mix points to try in order: the DJ's hot cues (E-F-G out, A-B-C in) when present, then the
        last full phrase of the outgoing track at the configured overlap, shrinking if it doesn't fit."""
        a_cues = _ordered_cues(hot_cue_beats(prev.item.analysis), "EFG")
        b_cues = _ordered_cues(hot_cue_beats(item.analysis), "ABC")
        j = b_cues.get("A", mix_in_beat(item.analysis))
        if j is None:
            return []
        in_end = b_cues.get("C") if b_cues.get("C", -1) > j else None
        in_mid = b_cues.get("B") if b_cues.get("B", -1) > j and (in_end is None or b_cues["B"] < in_end) else None
        in_label = "A" * ("A" in b_cues) + "B" * (in_mid is not None) + "C" * (in_end is not None)
        default_beats = self.settings.overlap_bars * 4

        candidates: list[MixPoints] = []
        k = a_cues.get("E")
        if k is not None:
            if a_cues.get("G", -1) > k:
                n_a = a_cues["G"] - k
            elif in_end is not None:
                n_a = in_end - j
            else:
                n_a = default_beats
            out_mid = a_cues.get("F") if k < a_cues.get("F", -1) < k + n_a else None
            out_label = "E" + "F" * (out_mid is not None) + "G" * (a_cues.get("G", -1) > k)
            candidates.append(MixPoints(k, n_a, j, out_mid, in_mid, in_end, _cue_label(out_label, in_label)))

        a_beats = prev.item.analysis.beats
        a_src = _beat_src(a_beats)
        downbeats = _downbeats(a_beats)
        for bars in sorted({b for b in (self.settings.overlap_bars, 8, 4) if b <= self.settings.overlap_bars}, reverse=True):
            n_a = bars * 4
            fitting = [
                d for d in downbeats if d + n_a < len(a_beats) and a_src[d + n_a] <= prev.item.audio.shape[0]
            ]
            if fitting:
                candidates.append(MixPoints(fitting[-1], n_a, j, None, in_mid, in_end, _cue_label("", in_label)))
        return candidates

    def _build_beatmatched(
        self,
        prev: Placement,
        item: LoadedTrack,
        points: MixPoints,
        t0: float,
        t1: float,
        anchor_out: float,
        rate: float,
    ) -> Placement:
        """Mix like a DJ on a mixer: the incoming track enters with its lows killed, climbing from
        `incoming_start_level` to full by the bass swap; at the swap the lows trade places; then the
        outgoing track, lows still killed, eases down to `outgoing_end_level` and is cut at the end."""
        a_src, b_src = _beat_src(prev.item.analysis.beats), _beat_src(item.analysis.beats)
        anchor_src = b_src[points.in_start]

        def b_time(beat: int) -> float:
            return anchor_out + (b_src[beat] - anchor_src) / rate

        beat = (t1 - t0) / points.out_beats
        if points.out_mid is not None:
            swap = prev.out_time_for_src(a_src[points.out_mid])
        elif points.in_mid is not None and t0 < b_time(points.in_mid) < t1:
            swap = b_time(points.in_mid)
        else:
            swap = prev.out_time_for_src(a_src[points.out_start + points.out_beats // 2])
        swap_start = swap - LOW_SWAP_BEATS * beat
        settings = self.settings

        prev.fade_out = Envelope((swap, t1 - CLICK_GUARD_FRAMES, t1), (1.0, settings.outgoing_end_level, 0.0))
        prev.lows_out = Envelope((swap_start, swap), (1.0, 0.0))
        prev.out_end = int(np.ceil(t1))
        prev.eq_regions = [*prev.eq_regions, (int(np.floor(swap_start)) - 1, prev.out_end)]

        ease_len = 0 if settings.hold_tempo else settings.ease_bars * 4 * beat
        if ease_len > 0:
            rate_points = [(t0, rate), (t1, rate), (t1 + ease_len, 1.0)]
        else:
            rate_points = [(t0, rate)]
        curve = RateCurve(rate_points, anchor_out=anchor_out, anchor_src=anchor_src)
        out_start = int(np.floor(t0))
        placement = Placement(
            item,
            curve,
            out_start,
            0,
            fade_in=Envelope((t0, t0 + CLICK_GUARD_FRAMES, swap), (0.0, settings.incoming_start_level, 1.0)),
            lows_in=Envelope((swap_start, swap), (0.0, 1.0)),
            switch_at=swap,
            overlap_in_end=t1,
            beatmatched=True,
            cues=points.cues,
            eq_regions=[(out_start, int(np.ceil(swap)) + 1)],
            align_with=prev,
        )
        placement.out_end = int(np.ceil(placement.out_time_for_src(item.audio.shape[0])))
        if settings.key_lock and abs(rate - 1) >= KEY_LOCK_MIN_TEMPO_CHANGE:
            placement.key_lock_end = int(np.ceil(rate_points[-1][0])) if curve.final_rate == 1.0 else placement.out_end
        return placement
