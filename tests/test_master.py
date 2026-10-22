import numpy as np
import pytest

from autoplayer.audio.decode import SAMPLE_RATE
from autoplayer.audio.master import CEILING_DB, HEADROOM_DB, MasterBus


def stream(master: MasterBus, audio: np.ndarray) -> np.ndarray:
    return np.concatenate([master.process(audio[i : i + 4096]) for i in range(0, audio.shape[0], 4096)])


def tone(amplitude: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.stack([wave, wave], axis=1)


def test_moderate_audio_only_gets_the_headroom_cut():
    audio = tone(0.8)
    out = stream(MasterBus(), audio)
    lag = audio.shape[0] - out.shape[0]
    np.testing.assert_allclose(out, audio[: out.shape[0]] * 10 ** (HEADROOM_DB / 20), atol=1e-5)
    assert lag < 0.01 * SAMPLE_RATE


def test_hot_overlapping_audio_is_held_under_the_ceiling_without_clipping():
    audio = tone(1.3) + tone(1.3)
    out = stream(MasterBus(), audio)
    ceiling = 10 ** (CEILING_DB / 20)
    assert np.max(np.abs(out)) <= ceiling + 1e-4
    steady = out[SAMPLE_RATE // 2 :, 0]
    spectrum = np.abs(np.fft.rfft(steady * np.hanning(len(steady))))
    freqs = np.fft.rfftfreq(len(steady), 1 / SAMPLE_RATE)
    harmonics = spectrum[(freqs > 300) & (freqs < 5000)].max()
    assert harmonics / spectrum.max() < 0.01


def test_reset_forgets_gain_reduction():
    master = MasterBus()
    stream(master, tone(3.0, 0.5))
    master.reset()
    out = stream(master, tone(0.5, 0.5))
    assert np.max(np.abs(out)) == pytest.approx(0.5 * 10 ** (HEADROOM_DB / 20), rel=0.01)
