from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoplayer.library.pdb import Library, PlaylistNode, Track
from autoplayer.library.search import search
from autoplayer.server.app import LibraryWatcher, create_app


def track(track_id: int, title: str, artist: str = "", key: str = "8A", bpm: float = 128.0) -> Track:
    return Track(track_id, title, artist, "", key, bpm, 200, Path(f"/x/{track_id}.mp3"), None)


TRACKS = [
    track(1, "Sweet Dreams (Dunisco Remix)", "Eurythmics", "5A", 135),
    track(2, "Monster (Extended Mix)", "Don Diablo", "7A", 145),
    track(3, "Closer (MOONLGHT Remix)", "The Chainsmokers", "4A", 128),
    track(4, "Get Busy", "Sean Paul", "4A", 126),
    track(5, "PHILLY ALRT REMIX - 9A - 134", "Travis Scott", "9A", 134),
    track(6, "Untitled", "", "", 124),
]


def titles(positions: list[int]) -> list[str]:
    return [TRACKS[p].title.split(" ")[0] for p in positions]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("", ["Sweet", "Monster", "Closer", "Get", "PHILLY", "Untitled"]),
        ("remix", ["Sweet", "Closer", "PHILLY"]),
        ("EURYTHMICS", ["Sweet"]),
        ("4A", ["Closer", "Get"]),
        ("4a", ["Closer", "Get"]),
        ("9A", ["PHILLY"]),
        ("Fm", ["Closer", "Get"]),
        ("128", ["Closer"]),
        ("125-135", ["Sweet", "Closer", "Get", "PHILLY"]),
        ("135-125", ["Sweet", "Closer", "Get", "PHILLY"]),
        ("remix 4A", ["Closer"]),
        ("nothing-like-this", []),
    ],
)
def test_search(query, expected):
    assert titles(search(TRACKS, query)) == expected


class StubEngine:
    def __init__(self):
        self.played = None
        self.queued = []

    def start(self):
        pass

    def close(self):
        pass

    def state(self):
        return {"status": "idle"}

    def play(self, tracks, index, playlist_id=None, playlist_name="", start_src=None, positions=None):
        self.played = ([t.title for t in tracks], index, playlist_id, playlist_name, positions)

    def queue_add(self, track, playlist_name="", position=None):
        self.queued.append((track.title, playlist_name))


@pytest.fixture
def client():
    playlist = PlaylistNode(id=7, name="Halloween", is_folder=False, parent_id=0, sort_order=0, track_ids=[4, 2, 3])
    library = Library(root=Path("/Volumes/TEST"), tracks={t.id: t for t in TRACKS}, playlists={7: playlist}, tree=[playlist])
    watcher = LibraryWatcher()
    watcher.refresh = lambda: None
    watcher.run = lambda: None
    watcher.library = library
    engine = StubEngine()
    with TestClient(create_app(engine, watcher)) as test_client:
        test_client.engine = engine
        yield test_client


def test_all_tracks_can_be_searched_and_sorted(client):
    body = client.get("/api/tracks", params={"query": "remix", "sort": "bpm", "descending": "true"}).json()
    assert body["name"] == "All tracks" and body["total"] == 6
    assert [t["title"].split(" ")[0] for t in body["tracks"]] == ["Sweet", "PHILLY", "Closer"]
    assert [t["position"] for t in body["tracks"]] == [0, 4, 2]


def test_playlists_can_be_searched(client):
    body = client.get("/api/playlists/7", params={"query": "4A"}).json()
    assert [t["title"] for t in body["tracks"]] == ["Get Busy", "Closer (MOONLGHT Remix)"]
    assert [t["position"] for t in body["tracks"]] == [0, 2]
    assert body["total"] == 3


def test_playing_a_search_result_plays_the_list_as_shown(client):
    client.post("/api/play", json={"playlist_id": None, "index": 1, "query": "4A", "sort": "bpm"})
    titles_played, index, playlist_id, label, positions = client.engine.played
    assert titles_played == ["Get Busy", "Closer (MOONLGHT Remix)"]
    assert (index, playlist_id, label, positions) == (1, None, "All tracks · “4A”", [3, 2])


def test_queueing_from_all_tracks_names_the_source(client):
    client.post("/api/queue", json={"track_id": 2, "playlist_id": None})
    assert client.engine.queued == [("Monster (Extended Mix)", "All tracks")]
