from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class RateCurve:
    """Playback rate as a piecewise-linear function of output sample time.

    The rate is held constant before the first point and after the last one. Source position is the
    exact integral of the rate, so it can be evaluated for any block of output samples independently,
    which keeps block boundaries seamless.
    """

    def __init__(self, points: Sequence[tuple[float, float]], anchor_out: float, anchor_src: float):
        if not points:
            raise ValueError("RateCurve needs at least one point")
        xs = np.array([p[0] for p in points], dtype=np.float64)
        if np.any(np.diff(xs) < 0):
            raise ValueError("RateCurve points must be sorted by time")
        self.xs = xs
        self.rates = np.array([p[1] for p in points], dtype=np.float64)
        segment_areas = np.diff(xs) * (self.rates[:-1] + self.rates[1:]) / 2
        self._cumulative = np.concatenate(([0.0], np.cumsum(segment_areas)))
        self._offset = anchor_src - self._integral(np.array([anchor_out], dtype=np.float64))[0]

    @property
    def final_rate(self) -> float:
        return float(self.rates[-1])

    def rate_at(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(t, self.xs, self.rates)

    def _integral(self, t: np.ndarray) -> np.ndarray:
        """Integral of the rate from xs[0] to t."""
        xs, rates, cumulative = self.xs, self.rates, self._cumulative
        idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 1)
        before = t < xs[0]
        dt = t - xs[idx]
        rate_t = np.interp(t, xs, rates)
        inside = cumulative[idx] + dt * (rates[idx] + rate_t) / 2
        after_last = idx == len(xs) - 1
        inside = np.where(after_last, cumulative[-1] + dt * rates[-1], inside)
        return np.where(before, (t - xs[0]) * rates[0], inside)

    def src_position(self, t: np.ndarray | float) -> np.ndarray:
        return self._offset + self._integral(np.asarray(t, dtype=np.float64))

    def out_time_for_src(self, src: float, lo: float, hi: float) -> float:
        """Invert src_position by bisection; rates are positive so the mapping is monotonic."""
        for _ in range(80):
            mid = (lo + hi) / 2
            if self.src_position(mid) < src:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def sample_cubic(audio: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Read `audio` (frames, channels) at fractional frame positions with Catmull-Rom interpolation.

    Positions outside the audio read as silence.
    """
    n = audio.shape[0]
    out = np.zeros((positions.shape[0], audio.shape[1]), dtype=np.float32)
    valid = (positions >= 0) & (positions <= n - 1)
    if n < 2 or not np.any(valid):
        return out
    pos = positions[valid]
    i1 = np.floor(pos).astype(np.int64)
    frac = (pos - i1).astype(np.float32)[:, None]
    i0 = np.clip(i1 - 1, 0, n - 1)
    i2 = np.clip(i1 + 1, 0, n - 1)
    i3 = np.clip(i1 + 2, 0, n - 1)
    p0, p1, p2, p3 = audio[i0], audio[i1], audio[i2], audio[i3]
    a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    c = -0.5 * p0 + 0.5 * p2
    out[valid] = ((a * frac + b) * frac + c) * frac + p1
    return out
