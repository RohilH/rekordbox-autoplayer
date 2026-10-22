from __future__ import annotations

from pathlib import Path

import av
import numpy as np

# Macs run their outputs (built-in speakers, displays, AirPods) at 48 kHz. Streaming at any other rate
# makes PortAudio convert on the fly and call back for a handful of frames at a time, which crackles.
SAMPLE_RATE = 48000
CHANNELS = 2


MAX_LEAD_IN_SECONDS = 0.5


def decode(path: Path) -> np.ndarray:
    """Decode any FFmpeg-readable file to float32 stereo at SAMPLE_RATE, shaped (frames, 2).

    FFmpeg trims encoder delay (LAME gapless info, AAC priming) and timestamps the first frame after
    it, but rekordbox's beat grids count from the untrimmed start. Padding the first frame's timestamp
    back in as silence keeps sample 0 at rekordbox's time 0; without it MP3 and M4A grids land
    25-48 ms late.
    """
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            if not chunks and frame.time is not None and 0 < frame.time < MAX_LEAD_IN_SECONDS:
                chunks.append(np.zeros((CHANNELS, round(frame.time * SAMPLE_RATE)), dtype=np.float32))
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks, axis=1).T, dtype=np.float32)
