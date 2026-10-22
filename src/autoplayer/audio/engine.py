from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from autoplayer.audio.clock import MixClock
from autoplayer.audio.decode import SAMPLE_RATE
from autoplayer.audio.keylock import ChunkedPitchCorrector
from autoplayer.audio.master import MasterBus
from autoplayer.audio.output import AudioOutput, BlockQueue
from autoplayer.audio.transition import LoadedTrack, MixSettings, Placement, Timeline, load_track
from autoplayer.audio.varispeed import sample_cubic
from autoplayer.library.pdb import Track

log = logging.getLogger(__name__)

BLOCK_FRAMES = 4096
BUFFER_TARGET_FRAMES = 2 * SAMPLE_RATE
PLAN_MARGIN_FRAMES = SAMPLE_RATE
SKIP_FADE_FRAMES = SAMPLE_RATE // 20
DECODE_CACHE_SIZE = 3
RETIME_LEAD_FRAMES = SAMPLE_RATE // 5
RETIME_CROSSFADE_FRAMES = 2048
INTERP_MARGIN = 2


@dataclass
class Session:
    generation: int
    playlist_id: int | None
    playlist_name: str
    tracks: list[Track]
    timeline: Timeline
    next_to_load: int
    clock: MixClock = field(default_factory=MixClock)
    feed_pos: int = 0
    corrector: ChunkedPitchCorrector | None = None
    version: int = 0
    start_src: float | None = None
    focus: Placement | None = None
    failed: list[int] = field(default_factory=list)

    @property
    def more_to_load(self) -> bool:
        return self.next_to_load < len(self.tracks)

    def tau(self, t: float) -> float:
        return float(self.clock.tau(t))


def _plain(value):
    """The state as plain Python values: timeline maths yields NumPy scalars, which JSON can't encode."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Engine:
    def __init__(self, device: int | str | None = None) -> None:
        self.settings = MixSettings()
        self.queue = BlockQueue()
        self.master = MasterBus()
        self.output = AudioOutput(self.queue, device)
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._session: Session | None = None
        self._generation = 0
        self.master_bpm: float | None = None
        self._cache: OrderedDict[int, LoadedTrack] = OrderedDict()
        self._running = True
        self._loading = False
        self._producer = threading.Thread(target=self._produce, name="producer", daemon=True)
        self._loader = threading.Thread(target=self._load, name="loader", daemon=True)

    def start(self) -> None:
        self.output.start()
        self._producer.start()
        self._loader.start()

    def close(self) -> None:
        with self._lock:
            self._running = False
            self._wake.notify_all()
        self.output.close()

    def play(
        self,
        tracks: list[Track],
        index: int,
        playlist_id: int | None = None,
        playlist_name: str = "",
        start_src: float | None = None,
    ) -> None:
        with self._lock:
            self._generation += 1
            speed = self.master_bpm / tracks[index].bpm if self.master_bpm and tracks[index].bpm else 1.0
            session = Session(
                generation=self._generation,
                playlist_id=playlist_id,
                playlist_name=playlist_name,
                tracks=tracks,
                timeline=Timeline(self.settings),
                next_to_load=index,
                clock=MixClock(speed),
                start_src=start_src,
            )
            self._session = session
            self.queue.reset(SKIP_FADE_FRAMES)
            self._restart_pipeline(session, 0)
            self.output.paused = False
            self._wake.notify_all()

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._session = None
            self.master.reset()
            self.queue.reset(SKIP_FADE_FRAMES)

    def pause(self) -> None:
        self.output.paused = True

    def resume(self) -> None:
        self.output.paused = False

    def skip(self) -> None:
        with self._lock:
            session = self._session
            if session is None:
                return
            current = self._current_placement(session)
            next_index = (current.item.index + 1) if current else session.next_to_load
            if next_index >= len(session.tracks):
                self.stop()
                return
            self.play(session.tracks, next_index, session.playlist_id, session.playlist_name)

    def seek(self, position_s: float) -> None:
        """Jump to `position_s` in the current track, on the planned mix's timeline.

        A position inside a transition lands where both tracks overlap, so they play tempo-matched with the
        crossfade gains of that moment. A position before the point where the mix brings this track in has
        no place on the timeline, so the track restarts from there on its own. Seeks never wait on the next
        track loading: if its transition isn't planned yet, it gets planned around wherever playback is.
        """
        with self._lock:
            session = self._session
            current = self._current_placement(session) if session else None
            if current is None:
                return

            src = min(max(position_s * SAMPLE_RATE, 0.0), current.item.audio.shape[0] - 1.0)
            if src < float(current.curve.src_position(current.out_start)):
                paused = self.output.paused
                self.play(session.tracks, current.item.index, session.playlist_id, session.playlist_name, src)
                self.output.paused = paused
                return
            target = min(current.out_time_for_src(src), current.out_end - 1)
            session.focus = current
            now = max(self.queue.played, 0)
            self.queue.reset(SKIP_FADE_FRAMES, position=now)
            session.clock.set(now, tau=target)
            self._restart_pipeline(session, now)
            self._wake.notify_all()

    def set_master_bpm(self, bpm: float | None) -> None:
        """Play the whole mix at `bpm`, or let tracks run at their own tempos again with None.

        The change lands a moment ahead of the play position, crossfaded in, by changing how fast the mix
        clock runs. Transitions not yet underway are replanned: with a master BPM every track holds the
        mix's tempo instead of easing back to its own.
        """
        with self._lock:
            mode_changed = (bpm is None) != (self.master_bpm is None)
            self.master_bpm = bpm
            self.settings = dataclasses.replace(self.settings, hold_tempo=bpm is not None)
            session = self._session
            if session is None:
                return
            session.timeline.settings = self.settings
            at = self._retime_point()
            if mode_changed:
                first_dropped = session.timeline.unplan_after(session.tau(at + PLAN_MARGIN_FRAMES))
                if first_dropped is not None:
                    session.next_to_load = first_dropped
            self._retime(session, at, speed=self._master_speed(session))

    def update_settings(self, **changes) -> MixSettings:
        with self._lock:
            key_lock_changed = "key_lock" in changes and changes["key_lock"] != self.settings.key_lock
            self.settings = dataclasses.replace(self.settings, **changes)
            session = self._session
            if session is not None:
                session.timeline.settings = self.settings
                if key_lock_changed:
                    self._retime(session, self._retime_point())
            return self.settings

    def _master_speed(self, session: Session) -> float:
        """Clock speed that puts the mix at the master BPM. Held tracks keep the tempo the latest placed
        track settles at, so that's the mix's own tempo."""
        if self.master_bpm is None or not session.timeline.placements:
            return 1.0
        latest = session.timeline.placements[-1]
        return self.master_bpm / (latest.item.bpm * latest.curve.final_rate)

    def _retime_point(self) -> int:
        played = max(self.queue.played, 0)
        return played + min(RETIME_LEAD_FRAMES, self.queue.queued)

    def _retime(self, session: Session, at: int, speed: float | None = None) -> None:
        """Rebuild the output from output time `at`: keep what's queued before it, crossfade into the new."""
        self.queue.splice(at - max(self.queue.played, 0), RETIME_CROSSFADE_FRAMES)
        session.clock.set(at, speed=speed)
        self._restart_pipeline(session, at)
        self._wake.notify_all()

    def _restart_pipeline(self, session: Session, at: int) -> None:
        session.version += 1
        session.feed_pos = at
        speed = session.clock.speed_at(at)
        needs_correction = self.settings.key_lock and abs(speed - 1) > 1e-6
        session.corrector = ChunkedPitchCorrector(speed, at) if needs_correction else None
        self.master.reset()

    def state(self) -> dict:
        return _plain(self._state())

    def _state(self) -> dict:
        with self._lock:
            session = self._session
            base = {
                "status": "idle",
                "paused": self.output.paused,
                "loading": self._loading,
                "settings": dataclasses.asdict(self.settings),
                "output_device": self.output.device_name,
                "playback_stats": self.queue.stats.to_json(),
                "master_bpm": self.master_bpm,
            }
            if session is None:
                return base
            played = max(self.queue.played, 0)
            tau = session.tau(played)
            speed = session.clock.speed_at(played)
            current = self._current_placement(session)
            base.update(
                status="paused" if self.output.paused else "playing",
                playlist_id=session.playlist_id,
                playlist_name=session.playlist_name,
                failed=session.failed,
            )
            if current is None:
                base["status"] = "loading" if session.more_to_load else "finished"
                return base
            track = session.tracks[current.item.index]
            following = self._placement_after(session, current)
            src_s = current.src_seconds_at
            overlapping = [p for p in session.timeline.placements if p.out_start <= tau < p.out_end]
            rate = float(current.curve.rate_at(tau)) * speed
            base.update(
                index=current.item.index,
                track=track.to_json(),
                position_s=current.src_seconds_at(tau),
                duration_s=current.item.audio.shape[0] / SAMPLE_RATE,
                tempo_change=rate - 1,
                bpm=current.item.bpm * rate,
                key_locked=current.key_lock_end is not None or session.corrector is not None,
                bass_swapped=current.lows_in is not None and tau >= current.lows_in.end,
                mixing=len(overlapping) > 1,
                beatmatched=current.beatmatched,
                next_index=following.item.index if following else None,
                next_beatmatched=following.beatmatched if following else None,
                cues=current.cues,
                next_cues=following.cues if following else None,
                mix_in_s=(session.clock.out_time(following.out_start) - played) / SAMPLE_RATE if following else None,
                play_start_s=src_s(current.out_start),
                play_end_s=src_s(current.out_end),
                mix_in_end_s=src_s(current.overlap_in_end) if current.overlap_in_end is not None else None,
                mix_out_start_s=src_s(following.out_start) if following and following.fade_in else None,
            )
            return base

    def _current_placement(self, session: Session) -> Placement | None:
        tau = session.tau(max(self.queue.played, 0))
        active = [p for p in session.timeline.placements if p.out_start <= tau < p.out_end]
        if not active:
            upcoming = [p for p in session.timeline.placements if p.out_start > tau]
            return upcoming[0] if upcoming else None
        if session.focus in active:
            return session.focus
        session.focus = None
        latest = active[-1]
        if latest.switch_at is not None and tau < latest.switch_at and len(active) > 1:
            return active[-2]
        return latest

    def _placement_after(self, session: Session, placement: Placement) -> Placement | None:
        placements = session.timeline.placements
        i = placements.index(placement)
        return placements[i + 1] if i + 1 < len(placements) else None

    def _produce(self) -> None:
        """Render ahead of the device. The rendering itself runs outside the engine lock so tempo changes,
        seeks and the UI never wait on it; anything a change made stale while it ran is thrown away."""
        while True:
            with self._lock:
                if not self._running:
                    return
                session = self._session
                if session is None or self.queue.queued >= BUFFER_TARGET_FRAMES:
                    self._wake.wait(timeout=0.02)
                    continue
                version, feed_pos, corrector = session.version, session.feed_pos, session.corrector
            block = self._render_block(session, feed_pos, corrector)
            with self._lock:
                if block is None or session is not self._session or session.version != version:
                    if block is None:
                        self._wake.wait(timeout=0.02)
                    continue
                session.feed_pos = feed_pos + BLOCK_FRAMES
                block = self.master.process(block)
                if block.shape[0]:
                    self.queue.push(block)
                self._prune(session)
                self._wake.notify_all()

    def _render_block(
        self, session: Session, feed_pos: int, corrector: ChunkedPitchCorrector | None
    ) -> np.ndarray | None:
        """The block of output at `feed_pos`, pitch-corrected when the mix clock runs off 1x, or None when
        there's nothing to render there yet."""
        if session.tau(feed_pos) >= session.timeline.end:
            return None
        if corrector is not None:
            return corrector.read(feed_pos, BLOCK_FRAMES, lambda s, n: self._raw(session, s, n))
        return self._raw(session, feed_pos, BLOCK_FRAMES)

    def _raw(self, session: Session, start: int, n: int) -> np.ndarray | None:
        """The mix over output frames [start, start + n), sampled along the mix clock, or None if part of
        that stretch of the timeline is still being prepared."""
        taus = session.clock.tau(np.arange(start, start + n, dtype=np.float64))
        lo = int(np.floor(taus[0])) - INTERP_MARGIN
        hi = int(np.ceil(taus[-1])) + INTERP_MARGIN + 1
        if any(p.pending_between(lo, hi) for p in list(session.timeline.placements)):
            return None
        return sample_cubic(session.timeline.render(lo, hi - lo), taus - lo)

    def _prune(self, session: Session) -> None:
        """Drop finished placements, keeping the one before the current track so a seek back into its transition works."""
        placements = session.timeline.placements
        finished = min(session.tau(session.feed_pos), session.tau(max(self.queue.played, 0)))
        while len(placements) > 2 and placements[1].out_end < finished:
            placements.pop(0)

    def _needs_track(self, session: Session) -> bool:
        if not session.more_to_load:
            return False
        placements = session.timeline.placements
        if not placements:
            return True
        tau = session.tau(max(self.queue.played, 0))
        return sum(1 for p in placements if p.out_end > tau) < 2

    def _load(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                session = self._session
                if session is None or not self._needs_track(session):
                    self._wake.wait(timeout=0.1)
                    continue
                index = session.next_to_load
                track = session.tracks[index]
                generation = session.generation
                self._loading = True
            try:
                item = self._load_cached(track, index)
            except Exception:
                log.exception("Could not load %s", track.audio_path)
                item = None
            with self._lock:
                self._loading = False
                session = self._session
                if session is None or session.generation != generation:
                    continue
                session.next_to_load = index + 1
                if item is None:
                    session.failed.append(index)
                    continue
                if session.timeline.placements:
                    not_before = int(session.tau(session.feed_pos + PLAN_MARGIN_FRAMES))
                    placement = session.timeline.append(item, not_before=not_before)
                else:
                    placement = session.timeline.append(item, start_src=session.start_src)
                pending = [p for p in session.timeline.placements if p.has_pending_work]
                self._wake.notify_all()
            started = time.perf_counter()
            for p in pending:
                if not p.cancelled:
                    p.prepare()
            if pending:
                log.info("Prepared %r's transition in %.1f s", placement.item.title, time.perf_counter() - started)
                with self._lock:
                    self._wake.notify_all()

    def _load_cached(self, track: Track, index: int) -> LoadedTrack:
        cached = self._cache.get(track.id)
        if cached is not None:
            self._cache.move_to_end(track.id)
            return dataclasses.replace(cached, index=index)
        item = load_track(track, index)
        self._cache[track.id] = item
        while len(self._cache) > DECODE_CACHE_SIZE:
            self._cache.popitem(last=False)
        return item
