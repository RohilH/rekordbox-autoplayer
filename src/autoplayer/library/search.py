from __future__ import annotations

import re
from dataclasses import dataclass

from autoplayer.library.keys import UNKNOWN_KEY, camelot_order
from autoplayer.library.pdb import Track

BPM_RANGE = re.compile(r"^(\d{2,3}(?:\.\d+)?)\s*-\s*(\d{2,3}(?:\.\d+)?)$")
BPM_VALUE = re.compile(r"^\d{2,3}(?:\.\d+)?$")
BPM_TOLERANCE = 0.5


@dataclass(frozen=True)
class Term:
    text: str
    key: tuple[int, int] | None
    bpm_low: float | None
    bpm_high: float | None

    def matches(self, track: Track) -> bool:
        if self.key is not None and camelot_order(track.key) == self.key:
            return True
        if self.bpm_low is not None and self.bpm_low <= track.bpm <= self.bpm_high:
            return True
        return any(self.text in field.casefold() for field in (track.title, track.artist, track.album))


def parse(query: str) -> list[Term]:
    """Split a search into terms that must all match. Each term matches title, artist or album text, and
    also a key (8A, Am) or a BPM (128, or a range like 120-128) when it looks like one."""
    terms = []
    for word in query.split():
        key = camelot_order(word)
        low = high = None
        if match := BPM_RANGE.match(word):
            low, high = sorted((float(match.group(1)), float(match.group(2))))
        elif BPM_VALUE.match(word):
            low, high = float(word) - BPM_TOLERANCE, float(word) + BPM_TOLERANCE
        terms.append(Term(word.casefold(), None if key == UNKNOWN_KEY else key, low, high))
    return terms


def search(tracks: list[Track], query: str) -> list[int]:
    """Positions of the tracks that match every term of `query` (all of them for an empty query)."""
    terms = parse(query)
    return [i for i, track in enumerate(tracks) if all(term.matches(track) for term in terms)]
