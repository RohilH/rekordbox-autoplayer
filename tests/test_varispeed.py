import numpy as np
import pytest

from autoplayer.audio.varispeed import RateCurve, sample_cubic


def test_constant_rate_maps_linearly():
    curve = RateCurve([(1000, 1.05)], anchor_out=1000, anchor_src=500)
    assert curve.src_position(1000) == pytest.approx(500)
    assert curve.src_position(2000) == pytest.approx(500 + 1050)
    assert curve.src_position(0) == pytest.approx(500 - 1050)


def test_ramp_position_is_the_integral_of_the_rate():
    curve = RateCurve([(0, 1.0), (100, 2.0)], anchor_out=0, anchor_src=0)
    assert curve.src_position(50) == pytest.approx(50 * 1.25)
    assert curve.src_position(100) == pytest.approx(150)
    assert curve.src_position(200) == pytest.approx(350)
    assert curve.src_position(-10) == pytest.approx(-10)


def test_anchor_can_sit_inside_a_ramp():
    curve = RateCurve([(0, 1.0), (100, 2.0)], anchor_out=100, anchor_src=1000)
    assert curve.src_position(0) == pytest.approx(850)


def test_positions_do_not_depend_on_block_boundaries():
    curve = RateCurve([(0, 1.08), (44100, 1.08), (88200, 1.0)], anchor_out=0, anchor_src=0)
    t = np.arange(0, 132300, dtype=np.float64)
    whole = curve.src_position(t)
    pieces = np.concatenate([curve.src_position(t[i : i + 4096]) for i in range(0, len(t), 4096)])
    np.testing.assert_array_equal(whole, pieces)
    assert curve.rate_at(132299) == pytest.approx(1.0)


def test_out_time_for_src_inverts_src_position():
    curve = RateCurve([(0, 0.95), (1000, 1.1)], anchor_out=0, anchor_src=0)
    for src in (10.0, 500.0, 1234.5, 5000.0):
        t = curve.out_time_for_src(src, 0, 10000)
        assert curve.src_position(t) == pytest.approx(src, abs=1e-6)


def test_cubic_reproduces_samples_at_integer_positions():
    audio = np.random.default_rng(0).standard_normal((100, 2)).astype(np.float32)
    out = sample_cubic(audio, np.arange(100, dtype=np.float64))
    np.testing.assert_allclose(out, audio, atol=1e-6)


def test_cubic_reads_outside_the_audio_as_silence():
    audio = np.ones((10, 2), dtype=np.float32)
    out = sample_cubic(audio, np.array([-1.0, 0.0, 9.0, 9.5]))
    np.testing.assert_array_equal(out[:, 0], [0, 1, 1, 0])


def test_faster_rate_raises_the_frequency():
    sr = 44100
    t = np.arange(sr * 2) / sr
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio = np.stack([tone, tone], axis=1)
    curve = RateCurve([(0, 1.05)], anchor_out=0, anchor_src=0)
    out = sample_cubic(audio, curve.src_position(np.arange(sr, dtype=np.float64)))[:, 0]
    crossings = np.count_nonzero(np.diff(np.signbit(out)))
    assert crossings / 2 == pytest.approx(440 * 1.05, rel=0.005)
