from __future__ import annotations

import numpy as np

from autoplayer.audio.decode import SAMPLE_RATE
from autoplayer.library.anlz import Beat

KICK_BAND_HZ = (30, 120)
SMOOTHING_FRAMES = 220
WINDOW_FRAMES = int(0.15 * SAMPLE_RATE)
MAX_OFFSET_S = 0.08
MIN_BEATS = 16


def kick_offset(audio: np.ndarray, beats: tuple[Beat, ...]) -> float:
    """Seconds from each grid beat to where the kick's low-band energy reaches half its peak.

    Averaging the envelope over every beat makes this stable within a track; it varies between tracks
    with kick shape and grid placement, so aligning these offsets rather than raw grid lines keeps
    kicks from flamming. Returns 0 when the estimate is unusable.
    """
    if audio.shape[0] == 0 or len(beats) < MIN_BEATS:
        return 0.0
    mono = audio.mean(axis=1, dtype=np.float64)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[0], 1 / SAMPLE_RATE)
    spectrum[(freqs < KICK_BAND_HZ[0]) | (freqs > KICK_BAND_HZ[1])] = 0
    low = np.fft.irfft(spectrum, n=mono.shape[0])
    envelope = np.convolve(low**2, np.ones(SMOOTHING_FRAMES) / SMOOTHING_FRAMES, mode="same")

    centers = [round(b.time_s * SAMPLE_RATE) for b in beats]
    centers = [c for c in centers if WINDOW_FRAMES <= c < envelope.shape[0] - WINDOW_FRAMES]
    if len(centers) < MIN_BEATS:
        return 0.0
    average = np.mean([envelope[c - WINDOW_FRAMES : c + WINDOW_FRAMES] for c in centers], axis=0)
    floor, peak = average[:WINDOW_FRAMES].min(), average.max()
    if peak <= floor:
        return 0.0
    rise = int(np.flatnonzero(average > floor + 0.5 * (peak - floor))[0])
    offset = (rise - WINDOW_FRAMES) / SAMPLE_RATE
    return offset if abs(offset) <= MAX_OFFSET_S else 0.0


ONSET_HOP = 128
ONSET_WINDOW = 1024
MAX_ONSET_LAG_S = 0.06


def onset_strength(audio: np.ndarray) -> np.ndarray:
    """Spectral flux: how much new energy appears across all frequencies at each hop, which peaks at the
    attack of each hit, where the ear places it."""
    mono = audio.mean(axis=1, dtype=np.float64) if audio.ndim == 2 else audio.astype(np.float64)
    if mono.shape[0] < ONSET_WINDOW * 2:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(mono, ONSET_WINDOW)[::ONSET_HOP] * np.hanning(ONSET_WINDOW)
    magnitude = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
    return np.maximum(np.diff(magnitude, axis=0), 0).sum(axis=1)


def onset_lag(a: np.ndarray, b: np.ndarray) -> float:
    """Seconds by which `b`'s hits land after `a`'s when both play together (negative: `b` is early),
    found by cross-correlating their onset strength. Returns 0 if they don't share a clear pulse."""
    fa, fb = onset_strength(a), onset_strength(b)
    n = min(fa.shape[0], fb.shape[0])
    if n < 64:
        return 0.0
    fa, fb = fa[:n] - fa[:n].mean(), fb[:n] - fb[:n].mean()
    reach = int(MAX_ONSET_LAG_S * SAMPLE_RATE / ONSET_HOP) + 1
    lags = np.arange(-reach, reach + 1)
    scores = np.array([np.dot(fa[max(0, -lag) : n - max(0, lag)], fb[max(0, lag) : n + min(0, lag)]) for lag in lags])
    best = int(np.argmax(scores))
    if best in (0, len(lags) - 1) or scores[best] <= 0:
        return 0.0
    left, mid, right = scores[best - 1 : best + 2]
    refine = 0.5 * (left - right) / (left - 2 * mid + right)
    return float((lags[best] + refine) * ONSET_HOP / SAMPLE_RATE)
