from pathlib import Path

from autoplayer.library.keys import camelot_order, sorted_positions
from autoplayer.library.pdb import Track


def track(key: str, bpm: float = 128.0) -> Track:
    return Track(0, "t", "a", "", key, bpm, 0, Path("x"), None)


def test_keys_sort_around_the_camelot_wheel_with_12b_last():
    keys = ["12B", "2A", "1B", "10A", "1A", "12A", "2B", "3A"]
    tracks = [track(k) for k in keys]
    assert [keys[i] for i in sorted_positions(tracks, "key")] == ["1A", "1B", "2A", "2B", "3A", "10A", "12A", "12B"]


def test_note_names_sort_with_their_camelot_keys():
    assert camelot_order("Am") == camelot_order("8A")
    assert camelot_order("C") == camelot_order("8B")


def test_tracks_without_a_key_stay_last_in_both_directions():
    keys = ["", "5A", "1A", "", "9B"]
    tracks = [track(k) for k in keys]
    assert [keys[i] for i in sorted_positions(tracks, "key")] == ["1A", "5A", "9B", "", ""]
    assert [keys[i] for i in sorted_positions(tracks, "key", descending=True)] == ["9B", "5A", "1A", "", ""]


def test_bpm_ties_keep_playlist_order_in_both_directions():
    tracks = [track("1A", bpm) for bpm in (126, 124, 126, 124)]
    assert sorted_positions(tracks, "bpm") == [1, 3, 0, 2]
    assert sorted_positions(tracks, "bpm", descending=True) == [0, 2, 1, 3]


def test_order_sort_can_reverse_the_playlist():
    tracks = [track("1A") for _ in range(4)]
    assert sorted_positions(tracks, "order") == [0, 1, 2, 3]
    assert sorted_positions(tracks, "order", descending=True) == [3, 2, 1, 0]
