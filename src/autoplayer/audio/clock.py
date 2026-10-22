from __future__ import annotations

import numpy as np


class MixClock:
    """Maps output time (frames since the session started) to mix time, the clock the timeline is planned on.

    Its speed is the master tempo relative to the mix's own, so changing it speeds the whole mix up or down
    together and transitions in progress stay aligned. A seek jumps it. Changes only ever happen at or
    after the current play position, so later segments are simply replaced.
    """

    def __init__(self, speed: float = 1.0) -> None:
        self._starts = [0.0]
        self._taus = [0.0]
        self._speeds = [speed]

    def set(self, at: float, tau: float | None = None, speed: float | None = None) -> None:
        """From output time `at`, run at `speed` from mix time `tau`. Left out, the speed is the latest one
        set, even if that was scheduled for later than `at`: it's the tempo the listener last asked for."""
        new_tau = float(self.tau(at)) if tau is None else tau
        new_speed = self._speeds[-1] if speed is None else speed
        keep = int(np.searchsorted(self._starts, at, side="left"))
        self._starts = self._starts[:keep] + [float(at)]
        self._taus = self._taus[:keep] + [new_tau]
        self._speeds = self._speeds[:keep] + [new_speed]

    def tau(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        idx = np.clip(np.searchsorted(self._starts, t, side="right") - 1, 0, len(self._starts) - 1)
        starts, taus, speeds = (np.asarray(a) for a in (self._starts, self._taus, self._speeds))
        return taus[idx] + speeds[idx] * (t - starts[idx])

    def speed_at(self, t: float) -> float:
        idx = max(int(np.searchsorted(self._starts, t, side="right")) - 1, 0)
        return self._speeds[idx]

    def out_time(self, tau: float) -> float:
        """Output time at which mix time `tau` plays, assuming the current speed holds."""
        return self._starts[-1] + (tau - self._taus[-1]) / self._speeds[-1]
