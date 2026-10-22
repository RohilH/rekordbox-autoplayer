from __future__ import annotations

import re

from autoplayer.library.pdb import Track

CAMELOT = re.compile(r"^\s*(\d{1,2})\s*([AB])\s*$", re.IGNORECASE)

# rekordbox can show keys as note names instead; these map them onto the Camelot wheel.
NOTE_NAMES = {
    "Abm": "1A", "G#m": "1A", "B": "1B",
    "Ebm": "2A", "D#m": "2A", "F#": "2B", "Gb": "2B",
    "Bbm": "3A", "A#m": "3A", "Db": "3B", "C#": "3B",
    "Fm": "4A", "Ab": "4B", "G#": "4B",
    "Cm": "5A", "Eb": "5B", "D#": "5B",
    "Gm": "6A", "Bb": "6B", "A#": "6B",
    "Dm": "7A", "F": "7B",
    "Am": "8A", "C": "8B",
    "Em": "9A", "G": "9B",
    "Bm": "10A", "D": "10B",
    "F#m": "11A", "Gbm": "11A", "A": "11B",
    "Dbm": "12A", "C#m": "12A", "E": "12B",
}

UNKNOWN_KEY = (99, 9)


def camelot_order(key: str) -> tuple[int, int]:
    """Sort position on the Camelot wheel: 1A, 1B, 2A, 2B … 12A, 12B, with unknown keys last."""
    match = CAMELOT.match(NOTE_NAMES.get(key.strip(), key))
    if not match:
        return UNKNOWN_KEY
    return int(match.group(1)), 0 if match.group(2).upper() == "A" else 1


SORTS = {
    "order": lambda track: (),
    "bpm": lambda track: (track.bpm,),
    "key": lambda track: camelot_order(track.key),
}


def sorted_positions(tracks: list[Track], sort: str = "order", descending: bool = False) -> list[int]:
    """Playlist positions of `tracks` in the requested order. Ties keep playlist order in either direction,
    and tracks with no known key stay at the end."""
    primary = SORTS[sort]
    sign = -1 if descending else 1

    def order(position: int) -> tuple:
        values = primary(tracks[position])
        unknown = sort == "key" and values == UNKNOWN_KEY
        if sort == "order":
            return (sign * position,)
        return (unknown, *(sign * v for v in values), position)

    return sorted(range(len(tracks)), key=order)
