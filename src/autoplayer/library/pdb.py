from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from kaitaistruct import BytesIO, KaitaiStream

from autoplayer.library._gen.rekordbox_pdb import RekordboxPdb

PDB_RELATIVE_PATH = Path("PIONEER/rekordbox/export.pdb")

PageType = RekordboxPdb.PageType


@dataclass(frozen=True)
class Track:
    id: int
    title: str
    artist: str
    album: str
    key: str
    bpm: float
    duration_s: int
    audio_path: Path
    anlz_path: Path | None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "key": self.key,
            "bpm": self.bpm,
            "duration_s": self.duration_s,
        }


@dataclass
class PlaylistNode:
    id: int
    name: str
    is_folder: bool
    parent_id: int
    sort_order: int
    children: list[PlaylistNode] = field(default_factory=list)
    track_ids: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "is_folder": self.is_folder,
            "track_count": len(self.track_ids),
            "children": [child.to_json() for child in self.children],
        }


@dataclass
class Library:
    root: Path
    tracks: dict[int, Track]
    playlists: dict[int, PlaylistNode]
    tree: list[PlaylistNode]

    def all_tracks(self) -> list[Track]:
        """Every track on the USB, in the order rekordbox added them."""
        return [self.tracks[track_id] for track_id in sorted(self.tracks)]

    def playlist_tracks(self, playlist_id: int) -> list[Track]:
        node = self.playlists[playlist_id]
        return [self.tracks[track_id] for track_id in node.track_ids if track_id in self.tracks]

    def find_playlist(self, name_or_id: str) -> PlaylistNode:
        if name_or_id.isdigit() and int(name_or_id) in self.playlists:
            return self.playlists[int(name_or_id)]
        matches = [p for p in self.playlists.values() if not p.is_folder and p.name == name_or_id]
        if not matches:
            matches = [
                p for p in self.playlists.values() if not p.is_folder and p.name.lower() == name_or_id.lower()
            ]
        if not matches:
            raise KeyError(f"No playlist named {name_or_id!r}")
        return matches[0]


def _text(device_sql_string) -> str:
    if device_sql_string is None:
        return ""
    return device_sql_string.body.text or ""


def _rows(db: RekordboxPdb, page_type: PageType) -> Iterator:
    for table in db.tables:
        if table.type != page_type:
            continue
        ref = table.first_page
        last_index = table.last_page.index
        seen: set[int] = set()
        while ref.index not in seen:
            seen.add(ref.index)
            page = ref.body
            if page.is_data_page:
                for group in page.row_groups:
                    for row in group.rows:
                        if row.present:
                            yield row.body
            if ref.index == last_index:
                break
            ref = page.next_page


def _usb_path(root: Path, stored: str) -> Path | None:
    if not stored:
        return None
    return root / stored.lstrip("/")


def load_library(root: Path) -> Library:
    data = (root / PDB_RELATIVE_PATH).read_bytes()
    db = RekordboxPdb(False, KaitaiStream(BytesIO(data)))

    artists = {row.id: _text(row.name) for row in _rows(db, PageType.artists)}
    albums = {row.id: _text(row.name) for row in _rows(db, PageType.albums)}
    keys = {row.id: _text(row.name) for row in _rows(db, PageType.keys)}

    tracks: dict[int, Track] = {}
    for row in _rows(db, PageType.tracks):
        tracks[row.id] = Track(
            id=row.id,
            title=_text(row.title),
            artist=artists.get(row.artist_id, ""),
            album=albums.get(row.album_id, ""),
            key=keys.get(row.key_id, ""),
            bpm=row.tempo / 100,
            duration_s=row.duration,
            audio_path=_usb_path(root, _text(row.file_path)),
            anlz_path=_usb_path(root, _text(row.analyze_path)),
        )

    playlists: dict[int, PlaylistNode] = {}
    for row in _rows(db, PageType.playlist_tree):
        playlists[row.id] = PlaylistNode(
            id=row.id,
            name=_text(row.name),
            is_folder=row.is_folder,
            parent_id=row.parent_id,
            sort_order=row.sort_order,
        )

    entries: dict[int, list[tuple[int, int]]] = {}
    for row in _rows(db, PageType.playlist_entries):
        entries.setdefault(row.playlist_id, []).append((row.entry_index, row.track_id))
    for playlist_id, items in entries.items():
        if playlist_id in playlists:
            playlists[playlist_id].track_ids = [track_id for _, track_id in sorted(items)]

    tree: list[PlaylistNode] = []
    for node in sorted(playlists.values(), key=lambda n: n.sort_order):
        parent = playlists.get(node.parent_id)
        (parent.children if parent else tree).append(node)

    return Library(root=root, tracks=tracks, playlists=playlists, tree=tree)
