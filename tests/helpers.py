from __future__ import annotations

import struct

import numpy as np

from autoplayer.audio.decode import SAMPLE_RATE
from autoplayer.audio.transition import LoadedTrack
from autoplayer.library.anlz import Analysis, Beat, Cue

CLICK_FRAMES = 64


def click_track(
    bpm: float, seconds: float, first_beat_s: float = 0.5, cues: tuple[Cue, ...] = (), index: int = 0
) -> LoadedTrack:
    """A stereo click on every beat, with a beat grid at millisecond resolution like rekordbox writes."""
    n = int(seconds * SAMPLE_RATE)
    audio = np.zeros((n, 2), dtype=np.float32)
    beats = []
    beat_s = 60 / bpm
    i = 0
    while True:
        time_ms = round((first_beat_s + i * beat_s) * 1000)
        start = round(time_ms * SAMPLE_RATE / 1000)
        if start + CLICK_FRAMES >= n:
            break
        audio[start : start + CLICK_FRAMES] = np.linspace(0.8, 0, CLICK_FRAMES, dtype=np.float32)[:, None]
        beats.append(Beat(time_s=time_ms / 1000, bar_pos=i % 4 + 1, bpm=bpm))
        i += 1
    return LoadedTrack(index=index, title=f"{bpm} BPM", bpm=bpm, audio=audio, analysis=Analysis(tuple(beats), cues))


def onsets(signal: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Frame indices where the absolute signal first rises above threshold, ignoring the rest of each click."""
    loud = np.abs(signal) > threshold
    rising = np.flatnonzero(loud[1:] & ~loud[:-1]) + 1
    if loud.size and loud[0]:
        rising = np.concatenate(([0], rising))
    return rising


def anlz_bytes(beats: list[tuple[int, int, int]], cues: list[tuple[int, int, int]]) -> bytes:
    """Build a minimal ANLZ .DAT: PQTZ beats (bar_pos, bpm*100, ms) and PCOB memory cues (hot_cue, type, ms)."""
    pqtz_body = struct.pack(">III", 0, 0x80000, len(beats)) + b"".join(struct.pack(">HHI", *b) for b in beats)
    pqtz = b"PQTZ" + struct.pack(">II", 24, 12 + len(pqtz_body)) + pqtz_body

    entries = b""
    for hot_cue, cue_type, time_ms in cues:
        entries += (
            b"PCPT"
            + struct.pack(">IIIII", 28, 56, hot_cue, 1, 0x10000)
            + struct.pack(">HH", 0xFFFF, 0xFFFF)
            + struct.pack(">B3x", cue_type)
            + struct.pack(">II", time_ms, 0xFFFFFFFF)
            + bytes(16)
        )
    pcob_body = struct.pack(">I2xHI", 0, len(cues), 0) + entries
    pcob = b"PCOB" + struct.pack(">II", 24, 12 + len(pcob_body)) + pcob_body

    header_len = 28
    body = pqtz + pcob
    return b"PMAI" + struct.pack(">II", header_len, header_len + len(body)) + bytes(header_len - 12) + body
