from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from kaitaistruct import BytesIO, KaitaiStream

from autoplayer.library._gen.rekordbox_anlz import RekordboxAnlz

CueType = RekordboxAnlz.CueEntryType


@dataclass(frozen=True)
class Beat:
    time_s: float
    bar_pos: int
    bpm: float


@dataclass(frozen=True)
class Cue:
    time_s: float
    hot_cue: int
    is_loop: bool
    comment: str = ""

    @property
    def is_memory_cue(self) -> bool:
        return self.hot_cue == 0 and not self.is_loop


@dataclass(frozen=True)
class Analysis:
    beats: tuple[Beat, ...]
    cues: tuple[Cue, ...]

    @property
    def memory_cues(self) -> list[Cue]:
        return sorted((c for c in self.cues if c.is_memory_cue), key=lambda c: c.time_s)


EMPTY_ANALYSIS = Analysis(beats=(), cues=())


TAG_BEAT_GRID = b"PQTZ"
TAG_CUES = b"PCOB"
TAG_CUES_EXTENDED = b"PCO2"


def _sections(path: Path) -> Iterator[tuple[bytes, KaitaiStream]]:
    """Walk the tagged sections by their headers so sections we don't use are never parsed."""
    data = path.read_bytes()
    if data[:4] != b"PMAI":
        return
    pos = struct.unpack_from(">I", data, 4)[0]
    while pos + 12 <= len(data):
        fourcc, _, len_tag = struct.unpack_from(">4sII", data, pos)
        if len_tag < 12:
            return
        yield fourcc, KaitaiStream(BytesIO(data[pos + 12 : pos + len_tag]))
        pos += len_tag


def _cues(tag) -> list[Cue]:
    return [
        Cue(
            time_s=entry.time / 1000,
            hot_cue=entry.hot_cue,
            is_loop=entry.type == CueType.loop,
            comment=(getattr(entry, "comment", None) or "").rstrip("\x00"),
        )
        for entry in tag.cues
        if entry.time != 0xFFFFFFFF
    ]


def load_analysis(dat_path: Path | None) -> Analysis:
    if dat_path is None or not dat_path.exists():
        return EMPTY_ANALYSIS

    beats: list[Beat] = []
    cues: list[Cue] = []
    for fourcc, stream in _sections(dat_path):
        if fourcc == TAG_BEAT_GRID:
            tag = RekordboxAnlz.BeatGridTag(stream)
            beats = [Beat(time_s=b.time / 1000, bar_pos=b.beat_number, bpm=b.tempo / 100) for b in tag.beats]
        elif fourcc == TAG_CUES:
            cues += _cues(RekordboxAnlz.CueTag(stream))

    ext_path = dat_path.with_suffix(".EXT")
    if ext_path.exists():
        extended = [
            cue
            for fourcc, stream in _sections(ext_path)
            if fourcc == TAG_CUES_EXTENDED
            for cue in _cues(RekordboxAnlz.CueExtendedTag(stream))
        ]
        if extended:
            cues = extended

    return Analysis(beats=tuple(beats), cues=tuple(sorted(cues, key=lambda c: c.time_s)))
