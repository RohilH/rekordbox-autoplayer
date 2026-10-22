from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pedalboard import time_stretch

from autoplayer.audio.decode import SAMPLE_RATE

UNISON_SEMITONES = 1e-4


def semitones_for(rate: float | np.ndarray) -> float | np.ndarray:
    return -12 * np.log2(rate)


def restore_pitch(varispeed_audio: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Undo the pitch change of audio that was played back at `rates`, keeping its timing.

    The varispeed playhead places every beat exactly; Rubber Band then shifts each part back by
    -12*log2(rate) semitones without changing the length, which is what master tempo does on a CDJ.
    """
    semitones = semitones_for(rates.astype(np.float64))
    if np.all(np.abs(semitones) < UNISON_SEMITONES):
        return varispeed_audio
    shifted = time_stretch(
        np.ascontiguousarray(varispeed_audio.T),
        SAMPLE_RATE,
        stretch_factor=1.0,
        pitch_shift_in_semitones=semitones,
    )
    out = np.ascontiguousarray(shifted.T, dtype=np.float32)
    if out.shape[0] < varispeed_audio.shape[0]:
        out = np.concatenate([out, np.zeros((varispeed_audio.shape[0] - out.shape[0], out.shape[1]), np.float32)])
    return out[: varispeed_audio.shape[0]]


class ChunkedPitchCorrector:
    """Holds the pitch of a stream sped up by `speed`, for the master tempo.

    pedalboard's streaming pitch shifter emits silence, so this corrects ahead of playback in overlapping
    offline chunks: each one gets a lead-in so Rubber Band settles before the part that's kept, and
    neighbours crossfade. The first chunk is short so a tempo change is heard almost at once.
    """

    FIRST_CHUNK_FRAMES = SAMPLE_RATE // 4
    CHUNK_FRAMES = 2 * SAMPLE_RATE
    PREROLL_FRAMES = int(0.3 * SAMPLE_RATE)
    POSTROLL_FRAMES = SAMPLE_RATE // 4
    OVERLAP_FRAMES = SAMPLE_RATE // 10

    def __init__(self, speed: float, start: int) -> None:
        self.speed = speed
        self._semitones = float(semitones_for(speed))
        self._buffer = np.zeros((0, 2), dtype=np.float32)
        self._buffer_start = start
        self._tail: np.ndarray | None = None

    @property
    def _ready_end(self) -> int:
        return self._buffer_start + self._buffer.shape[0]

    def read(self, start: int, n: int, render: Callable[[int, int], np.ndarray | None]) -> np.ndarray | None:
        """Corrected audio for output frames [start, start + n), rendering the raw stream with `render`
        (which returns None while the timeline isn't ready there)."""
        while self._ready_end < start + n:
            if not self._extend(render):
                return None
        offset = start - self._buffer_start
        block = self._buffer[offset : offset + n]
        self._buffer = self._buffer[offset + n :]
        self._buffer_start = start + n
        return block

    def _extend(self, render: Callable[[int, int], np.ndarray | None]) -> bool:
        begin = self._ready_end
        length = self.FIRST_CHUNK_FRAMES if self._tail is None else self.CHUNK_FRAMES
        overlap = self.OVERLAP_FRAMES
        raw = render(begin - self.PREROLL_FRAMES, self.PREROLL_FRAMES + length + overlap + self.POSTROLL_FRAMES)
        if raw is None:
            return False
        shifted = time_stretch(
            np.ascontiguousarray(raw.T), SAMPLE_RATE, stretch_factor=1.0, pitch_shift_in_semitones=self._semitones
        )
        body = np.ascontiguousarray(shifted.T[self.PREROLL_FRAMES : self.PREROLL_FRAMES + length + overlap], dtype=np.float32)
        if self._tail is not None:
            fade_in = np.linspace(0, 1, overlap, dtype=np.float32)[:, None]
            body[:overlap] = self._tail * (1 - fade_in) + body[:overlap] * fade_in
        self._buffer = np.concatenate([self._buffer, body[:length]])
        self._tail = body[length:]
        return True
