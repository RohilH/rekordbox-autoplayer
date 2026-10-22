from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from autoplayer.audio.decode import SAMPLE_RATE

LOW_CUTOFF_HZ = 180.0
_LOWPASS = butter(4, LOW_CUTOFF_HZ, btype="low", fs=SAMPLE_RATE, output="sos")


def split_lows(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split (frames, channels) audio into lows and everything else, like a mixer's low-EQ kill.

    The low-pass runs forwards and backwards, so it has no phase shift and the rest is simply the
    remainder: the two bands always add back to exactly the input, which lets a band be faded in or out
    without changing anything while it's at full level.
    """
    lows = sosfiltfilt(_LOWPASS, audio, axis=0).astype(np.float32)
    return lows, (audio - lows).astype(np.float32)
