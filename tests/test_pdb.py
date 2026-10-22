"""Checks the PDB reader against a real rekordbox export.

Copy `PIONEER/rekordbox/export.pdb` from a USB into tests/fixtures/usb/PIONEER/rekordbox/, or point
AUTOPLAYER_TEST_USB at a mounted USB.
"""

import os
from pathlib import Path

import pytest

from autoplayer.library.pdb import PDB_RELATIVE_PATH, load_library

FIXTURE_ROOT = Path(os.environ.get("AUTOPLAYER_TEST_USB", Path(__file__).parent / "fixtures" / "usb"))

pytestmark = pytest.mark.skipif(
    not (FIXTURE_ROOT / PDB_RELATIVE_PATH).is_file(), reason="no rekordbox export.pdb fixture"
)


def test_reads_tracks_and_playlists():
    library = load_library(FIXTURE_ROOT)
    assert library.tracks
    assert all(t.audio_path is not None for t in library.tracks.values())
    playlists = [p for p in library.playlists.values() if not p.is_folder]
    assert playlists
    for playlist in playlists:
        assert len(library.playlist_tracks(playlist.id)) == len(playlist.track_ids)
