import dataclasses

import numpy as np
import pytest

from autoplayer.audio.decode import SAMPLE_RATE
from autoplayer.audio.master import CEILING_DB, MasterBus
from autoplayer.audio.onsets import kick_offset
from autoplayer.audio.transition import CLICK_GUARD_FRAMES, LOW_SWAP_BEATS, MixSettings, Placement, Timeline, mix_in_beat
from autoplayer.audio.varispeed import sample_cubic
from autoplayer.library.anlz import Cue
from tests.helpers import click_track, onsets

FIVE_MS = 0.005 * SAMPLE_RATE


def ungained(placement: Placement, start: int, end: int) -> np.ndarray:
    t = np.arange(start, end, dtype=np.float64)
    return sample_cubic(placement.item.audio, placement.curve.src_position(t))[:, 0]


def assert_aligned(a_onsets: np.ndarray, b_onsets: np.ndarray, tolerance: float = FIVE_MS, edge: int = SAMPLE_RATE // 4):
    """Every hit of b lines up with one of a, ignoring hits too near the ends of the window to have a partner."""
    inner = b_onsets[(b_onsets > a_onsets.min() + edge) & (b_onsets < a_onsets.max() - edge)]
    assert len(inner) >= 50
    nearest = np.min(np.abs(inner[:, None] - a_onsets[None, :]), axis=1)
    assert np.max(nearest) < tolerance


def mix(a, b, **settings):
    timeline = Timeline(MixSettings(**settings))
    first = timeline.append(a)
    second = timeline.append(b)
    return timeline, first, second


def test_beatmatched_clicks_coincide_through_the_overlap():
    _, a, b = mix(click_track(124, 120), click_track(128, 120, first_beat_s=0.31, index=1))

    assert b.beatmatched
    assert b.curve.rate_at(b.out_start) == pytest.approx(124 / 128, rel=1e-3)
    start, end = b.out_start, a.out_end
    assert (end - start) / SAMPLE_RATE == pytest.approx(16 * 4 * 60 / 124, rel=1e-3)

    assert_aligned(onsets(ungained(a, start, end)), onsets(ungained(b, start, end)))


def test_incoming_track_eases_back_to_its_own_tempo():
    _, a, b = mix(click_track(124, 120), click_track(128, 120, index=1), ease_bars=8)

    ease_end = int(b.curve.xs[-1])
    assert b.curve.rate_at(ease_end + 1) == pytest.approx(1.0)
    tail = onsets(ungained(b, ease_end, ease_end + 10 * SAMPLE_RATE))
    assert np.median(np.diff(tail)) / SAMPLE_RATE == pytest.approx(60 / 128, abs=0.001)


def test_zero_ease_holds_the_outgoing_tempo():
    _, _, b = mix(click_track(124, 120), click_track(128, 120, index=1), ease_bars=0)
    assert b.curve.final_rate == pytest.approx(124 / 128, rel=1e-3)


def test_half_time_tracks_are_matched_at_double_time():
    _, _, b = mix(click_track(140, 120), click_track(70, 240, index=1))
    assert b.beatmatched
    assert b.curve.rate_at(b.out_start) == pytest.approx(1.0, abs=0.01)


def test_tempo_gap_beyond_pitch_range_falls_back_to_a_timed_crossfade():
    _, a, b = mix(click_track(120, 120), click_track(140, 120, index=1), fade_seconds=6)
    assert not b.beatmatched
    assert b.fade_in.end - b.fade_in.start == pytest.approx(6 * SAMPLE_RATE)
    assert b.curve.final_rate == 1.0
    assert a.out_end == pytest.approx(b.fade_in.end, abs=1)


def test_tracks_without_a_beat_grid_fall_back_to_a_timed_crossfade():
    a = click_track(124, 120)
    b = click_track(128, 120, index=1)
    b.analysis = type(b.analysis)(beats=(), cues=())
    _, _, placed = mix(a, b)
    assert not placed.beatmatched
    assert placed.fade_in is not None


def test_gapless_mode_starts_the_next_track_where_the_last_one_goes_silent():
    _, a, b = mix(click_track(124, 60), click_track(128, 60, first_beat_s=1.0, index=1), mode="off")
    assert b.fade_in is None
    assert a.out_end == b.out_start
    assert b.curve.src_position(b.out_start) == pytest.approx(1.0 * SAMPLE_RATE, abs=1)


def test_mix_in_snaps_the_first_memory_cue_to_the_nearest_downbeat():
    track = click_track(120, 120, first_beat_s=0.5, cues=(Cue(time_s=30.4, hot_cue=0, is_loop=False),))
    beat = track.analysis.beats[mix_in_beat(track.analysis)]
    assert beat.bar_pos == 1
    assert beat.time_s == pytest.approx(30.5)


def test_mix_in_defaults_to_the_first_downbeat():
    track = click_track(120, 60)
    assert mix_in_beat(track.analysis) == 0


def test_mastered_mix_stays_under_the_ceiling():
    timeline, _, b = mix(click_track(124, 120), click_track(128, 120, index=1))
    master = MasterBus()
    audio = np.concatenate([master.process(timeline.render(t, 4096)) for t in range(b.out_start, b.out_start + 20 * SAMPLE_RATE, 4096)])
    assert np.max(np.abs(audio)) <= 10 ** (CEILING_DB / 20) + 1e-4


def test_kick_offsets_are_aligned_instead_of_grid_lines():
    late = click_track(128, 120, index=1)
    shift = int(0.02 * SAMPLE_RATE)
    late.audio = np.concatenate([np.zeros((shift, 2), dtype=np.float32), late.audio[:-shift]])
    a = click_track(124, 120)
    a.kick_offset_s = kick_offset(a.audio, a.analysis.beats)
    late.kick_offset_s = kick_offset(late.audio, late.analysis.beats)
    assert late.kick_offset_s - a.kick_offset_s == pytest.approx(0.02, abs=0.001)

    _, a, b = mix(a, late)
    assert_aligned(onsets(ungained(a, b.out_start, a.out_end)), onsets(ungained(b, b.out_start, a.out_end)))


def solo(placement: Placement, start: int, end: int) -> np.ndarray:
    """What a placement plays over [start, end), ignoring its fades."""
    fades = placement.fade_in, placement.fade_out
    placement.fade_in = placement.fade_out = None
    buffer = np.zeros((end - start, 2), dtype=np.float32)
    placement.render_into(buffer, start)
    placement.fade_in, placement.fade_out = fades
    return buffer[:, 0]


def peak_hz(signal: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return float(np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)[spectrum.argmax()])


def with_tone(track, hz=440.0):
    t = np.arange(track.audio.shape[0]) / SAMPLE_RATE
    track.audio = track.audio + (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)[:, None]
    return track


def test_key_lock_holds_pitch_while_the_tempo_is_matched():
    timeline, a, b = mix(click_track(124, 120), with_tone(click_track(128, 120, index=1)))
    assert b.has_pending_work
    timeline.prepare()

    overlap = solo(b, b.out_start + SAMPLE_RATE, a.out_end - SAMPLE_RATE)
    assert peak_hz(overlap) == pytest.approx(440, abs=1)
    raw = ungained(b, b.out_start + SAMPLE_RATE, a.out_end - SAMPLE_RATE)
    assert peak_hz(raw) == pytest.approx(440 * 124 / 128, abs=1)


def test_key_locked_clicks_still_coincide():
    timeline, a, b = mix(click_track(124, 120), click_track(128, 120, index=1))
    timeline.prepare()
    start, end = b.out_start + SAMPLE_RATE, a.out_end - SAMPLE_RATE
    a_onsets = onsets(solo(a, start, end), threshold=0.2)
    b_onsets = onsets(solo(b, start, end), threshold=0.2)
    assert len(b_onsets) >= 55
    nearest = np.min(np.abs(b_onsets[:, None] - a_onsets[None, :]), axis=1)
    assert np.max(nearest) < FIVE_MS


def test_key_lock_splices_back_to_the_original_audio_after_the_ease():
    timeline, _, b = mix(click_track(124, 120), with_tone(click_track(128, 120, index=1)))
    timeline.prepare()
    after = b.key_lock_end + 2 * SAMPLE_RATE
    np.testing.assert_allclose(solo(b, after, after + 1000), ungained(b, after, after + 1000), atol=1e-6)


def test_matching_tempos_need_no_key_lock():
    _, _, b = mix(click_track(128, 120), click_track(128, 120, index=1))
    assert b.key_lock_end is None


def test_key_lock_can_be_turned_off():
    _, _, b = mix(click_track(124, 120), click_track(128, 120, index=1), key_lock=False)
    assert b.key_lock_end is None


def with_hot_cues(track, **letters_to_beats):
    cues = tuple(
        Cue(time_s=track.analysis.beats[beat].time_s, hot_cue=ord(letter) - ord("A") + 1, is_loop=False)
        for letter, beat in letters_to_beats.items()
    )
    track.analysis = dataclasses.replace(track.analysis, cues=cues)
    return track


def out_time_of_beat(placement: Placement, beat: int) -> float:
    return placement.out_time_for_src(placement.item.analysis.beats[beat].time_s * SAMPLE_RATE)


def test_hot_cues_set_the_transition_points():
    outgoing = with_hot_cues(click_track(124, 150), E=160, F=192, G=224)
    incoming = with_hot_cues(click_track(128, 150, index=1), A=32, B=64, C=96)
    _, a, b = mix(outgoing, incoming)

    assert b.beatmatched
    e, f, g = (out_time_of_beat(a, beat) for beat in (160, 192, 224))
    assert out_time_of_beat(b, 32) == pytest.approx(e, abs=FIVE_MS)
    assert b.out_start == pytest.approx(e, abs=1)
    assert b.switch_at == pytest.approx(f)
    assert a.out_end == pytest.approx(g, abs=1)

    a_onsets = onsets(ungained(a, b.out_start, a.out_end))
    b_onsets = onsets(ungained(b, b.out_start, a.out_end))
    nearest = np.min(np.abs(b_onsets[:, None] - a_onsets[None, :]), axis=1)
    assert len(b_onsets) >= 63
    assert np.max(nearest) < FIVE_MS


def mixer_levels(placement: Placement, t: float) -> tuple[float, float]:
    return float(placement.gain(np.array([t]))[0]), float(placement.low_level(np.array([t]))[0])


def test_transition_follows_the_mixer_moves():
    _, a, b = mix(click_track(128, 150), click_track(128, 150, index=1))
    start, end = b.out_start, a.out_end
    beat = (end - start) / 64
    swap = b.switch_at
    assert swap == pytest.approx(start + 32 * beat, abs=2)

    just_in = start + CLICK_GUARD_FRAMES
    assert mixer_levels(b, just_in) == pytest.approx((0.5, 0.0))
    assert mixer_levels(a, just_in) == pytest.approx((1.0, 1.0))
    assert mixer_levels(b, start + 16 * beat)[0] == pytest.approx(0.75, abs=0.01)

    before_swap = swap - LOW_SWAP_BEATS * beat - 1
    assert mixer_levels(b, before_swap)[1] == 0.0
    assert mixer_levels(a, before_swap) == pytest.approx((1.0, 1.0))
    assert mixer_levels(b, swap) == pytest.approx((1.0, 1.0))
    assert mixer_levels(a, swap) == pytest.approx((1.0, 0.0))

    assert mixer_levels(a, end - CLICK_GUARD_FRAMES) == pytest.approx((0.8, 0.0))
    assert mixer_levels(a, end)[0] == 0.0
    assert mixer_levels(b, end + SAMPLE_RATE) == pytest.approx((1.0, 1.0))


def with_bass_and_lead(track, bass_hz=60.0, lead_hz=2000.0):
    t = np.arange(track.audio.shape[0]) / SAMPLE_RATE
    tones = 0.3 * np.sin(2 * np.pi * bass_hz * t) + 0.3 * np.sin(2 * np.pi * lead_hz * t)
    track.audio = tones.astype(np.float32)[:, None].repeat(2, axis=1)
    return track


def band_db(signal: np.ndarray, hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) / (len(signal) / 4)
    freqs = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
    return float(20 * np.log10(spectrum[np.argmin(np.abs(freqs - hz))] / 0.3))


def test_lows_are_killed_and_swapped():
    timeline, a, b = mix(with_bass_and_lead(click_track(128, 150)), with_bass_and_lead(click_track(128, 150, index=1)))
    timeline.prepare()
    start, swap, end = b.out_start, int(b.switch_at), a.out_end
    first_half = slice(start + SAMPLE_RATE, swap - SAMPLE_RATE)
    second_half = slice(swap + SAMPLE_RATE // 2, end - SAMPLE_RATE // 2)

    incoming_first = solo(b, first_half.start, first_half.stop)
    outgoing_first = solo(a, first_half.start, first_half.stop)
    assert band_db(incoming_first, 60) < -40
    assert band_db(incoming_first, 2000) == pytest.approx(0, abs=0.5)
    assert band_db(outgoing_first, 60) == pytest.approx(0, abs=0.5)

    incoming_second = solo(b, second_half.start, second_half.stop)
    outgoing_second = solo(a, second_half.start, second_half.stop)
    assert band_db(incoming_second, 60) == pytest.approx(0, abs=0.5)
    assert band_db(outgoing_second, 60) < -40
    assert band_db(outgoing_second, 2000) == pytest.approx(0, abs=0.5)


def test_band_split_is_transparent_outside_the_swap():
    timeline, a, b = mix(with_bass_and_lead(click_track(128, 150)), with_bass_and_lead(click_track(128, 150, index=1)))
    timeline.prepare()
    after_swap = int(np.ceil(b.switch_at)) + 1
    np.testing.assert_allclose(solo(b, after_swap, after_swap + 1000), ungained(b, after_swap, after_swap + 1000), atol=1e-5)
    before_swap = int(np.floor(a.lows_out.start)) - 1
    np.testing.assert_allclose(solo(a, before_swap - 1000, before_swap), ungained(a, before_swap - 1000, before_swap), atol=1e-5)


def test_incoming_cues_are_used_without_outgoing_cues():
    incoming = with_hot_cues(click_track(128, 150, index=1), A=32, B=64, C=96)
    _, a, b = mix(click_track(124, 150), incoming)
    assert b.beatmatched
    assert b.src_seconds_at(b.fade_in.start) == pytest.approx(incoming.analysis.beats[32].time_s, abs=0.005)
    last_downbeat_phrase = a.out_end - b.out_start
    assert last_downbeat_phrase / SAMPLE_RATE == pytest.approx(64 * 60 / 124, rel=1e-3)


def test_outro_without_an_end_cue_uses_the_intro_length():
    outgoing = with_hot_cues(click_track(128, 150), E=160)
    incoming = with_hot_cues(click_track(128, 150, index=1), A=32, C=64)
    _, a, b = mix(outgoing, incoming)
    assert (a.out_end - b.out_start) / SAMPLE_RATE == pytest.approx(32 * 60 / 128, rel=1e-3)


def test_cues_that_do_not_fit_fall_back_to_the_last_phrase():
    outgoing = with_hot_cues(click_track(124, 150), E=300)
    _, a, b = mix(outgoing, click_track(128, 150, index=1))
    assert b.beatmatched
    assert (a.out_end - b.out_start) / SAMPLE_RATE == pytest.approx(64 * 60 / 124, rel=1e-3)


def test_timed_crossfade_starts_the_next_track_at_its_a_cue():
    incoming = with_hot_cues(click_track(140, 120, index=1), A=32)
    _, _, b = mix(click_track(120, 120), incoming)
    assert not b.beatmatched
    assert b.src_seconds_at(b.fade_in.start) == pytest.approx(incoming.analysis.beats[32].time_s, abs=0.001)


def test_auto_generated_cues_are_ignored():
    outgoing = click_track(124, 150)
    beats = outgoing.analysis.beats
    outgoing.analysis = dataclasses.replace(
        outgoing.analysis,
        cues=(Cue(time_s=beats[100].time_s, hot_cue=5, is_loop=False, comment="CUE(Auto)"),),
    )
    _, a, b = mix(outgoing, click_track(128, 150, index=1))
    assert b.cues == ""
    assert (a.out_end - b.out_start) / SAMPLE_RATE == pytest.approx(64 * 60 / 124, rel=1e-3)


def test_out_of_order_cues_are_ignored_for_that_side():
    incoming = with_hot_cues(click_track(128, 150, index=1), A=64, B=32, C=96)
    outgoing = with_hot_cues(click_track(124, 150), E=160, G=224)
    _, a, b = mix(outgoing, incoming)
    assert b.cues == "EG out"
    assert b.src_seconds_at(b.fade_in.start) == pytest.approx(incoming.analysis.beats[0].time_s, abs=0.005)


def test_bass_swap_falls_on_the_outgoing_f_cue():
    outgoing = with_hot_cues(click_track(128, 150), E=160, F=176, G=224)
    incoming = with_hot_cues(click_track(128, 150, index=1), A=0, B=32, C=64)
    _, a, b = mix(outgoing, incoming)
    assert b.switch_at == pytest.approx(out_time_of_beat(a, 176))


def test_bass_swap_uses_the_incoming_b_cue_without_an_f_cue():
    outgoing = with_hot_cues(click_track(128, 150), E=160, G=224)
    incoming = with_hot_cues(click_track(128, 150, index=1), A=0, B=24, C=64)
    _, a, b = mix(outgoing, incoming)
    assert b.switch_at == pytest.approx(out_time_of_beat(b, 24), abs=FIVE_MS)


def test_attacks_are_lined_up_even_when_the_kick_estimates_are_wrong():
    early = click_track(128, 150, index=1)
    shift = int(0.025 * SAMPLE_RATE)
    early.audio = np.concatenate([early.audio[shift:], np.zeros((shift, 2), dtype=np.float32)])
    timeline, a, b = mix(click_track(124, 150), early)
    timeline.prepare()
    assert_aligned(
        onsets(ungained(a, b.out_start, a.out_end)), onsets(ungained(b, b.out_start, a.out_end)), 0.002 * SAMPLE_RATE
    )
