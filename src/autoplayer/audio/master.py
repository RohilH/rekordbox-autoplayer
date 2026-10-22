from __future__ import annotations

import numpy as np
from pedalboard import BrickwallLimiter

from autoplayer.audio.decode import SAMPLE_RATE

HEADROOM_DB = -3.0
CEILING_DB = -1.0
RELEASE_MS = 100.0


class MasterBus:
    """Final gain stage: headroom, then a lookahead brick-wall limiter.

    Club masters decode with peaks well above full scale, and two of them overlapping in a mix sum
    higher still. The headroom keeps a single hot track clear of the ceiling; the limiter turns the level
    down smoothly when an overlap goes over instead of clipping the waveform, which is what crackles.

    It keeps state between blocks, so feed it the stream in order and `reset` it on any jump. The
    lookahead holds back a few milliseconds after a reset, so outputs can be shorter than inputs.
    """

    def __init__(self) -> None:
        self._gain = np.float32(10 ** (HEADROOM_DB / 20))
        self._limiter = BrickwallLimiter(ceiling_db=CEILING_DB, release_ms=RELEASE_MS, true_peak=True)

    def process(self, block: np.ndarray) -> np.ndarray:
        limited = self._limiter.process(np.ascontiguousarray((block * self._gain).T), SAMPLE_RATE, reset=False)
        return np.ascontiguousarray(limited.T, dtype=np.float32)

    def reset(self) -> None:
        self._limiter.reset()
