from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from autoplayer.audio.engine import Engine
from autoplayer.library.pdb import Library, load_library
from autoplayer.library.usb import find_usb_roots

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
USB_POLL_SECONDS = 3.0
STATE_PUSH_SECONDS = 0.25


class LibraryWatcher:
    """Keeps the library of the first mounted rekordbox USB loaded, or of a fixed path when given one."""

    def __init__(self, fixed_root: Path | None = None) -> None:
        self.fixed_root = fixed_root
        self.library: Library | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def refresh(self) -> None:
        roots = [self.fixed_root] if self.fixed_root else find_usb_roots()
        root = roots[0] if roots else None
        with self._lock:
            if root is None:
                self.library, self.error = None, None
                return
            if self.library is not None and self.library.root == root:
                return
        try:
            library, error = load_library(root), None
        except Exception as exc:
            log.exception("Could not read rekordbox export on %s", root)
            library, error = None, f"Could not read {root}: {exc}"
        with self._lock:
            self.library, self.error = library, error

    def run(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(USB_POLL_SECONDS)

    def stop(self) -> None:
        self._stop.set()


class PlayRequest(BaseModel):
    playlist_id: int
    index: int = 0


class MasterBpmRequest(BaseModel):
    bpm: float | None = Field(default=None, ge=40, le=300)


class QueueAddRequest(BaseModel):
    track_id: int
    playlist_id: int | None = None
    position: int | None = Field(default=None, ge=0)


class QueueMoveRequest(BaseModel):
    position: int = Field(ge=0)


class SeekRequest(BaseModel):
    position_s: float = Field(ge=0)


class SettingsUpdate(BaseModel):
    mode: Literal["off", "timed", "beatmatch"] | None = None
    overlap_bars: int | None = Field(default=None, ge=1, le=64)
    ease_bars: int | None = Field(default=None, ge=0, le=64)
    max_tempo_change: float | None = Field(default=None, gt=0, le=0.5)
    fade_seconds: float | None = Field(default=None, gt=0, le=60)
    key_lock: bool | None = None


def create_app(engine: Engine, watcher: LibraryWatcher) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        watcher.refresh()
        threading.Thread(target=watcher.run, name="usb-watcher", daemon=True).start()
        engine.start()
        yield
        watcher.stop()
        engine.close()

    app = FastAPI(lifespan=lifespan)

    def require_library() -> Library:
        if watcher.library is None:
            raise HTTPException(status_code=404, detail=watcher.error or "No rekordbox USB found")
        return watcher.library

    @app.get("/api/library")
    def get_library() -> dict:
        library = watcher.library
        if library is None:
            return {"connected": False, "error": watcher.error}
        return {
            "connected": True,
            "root": str(library.root),
            "name": library.root.name,
            "playlists": [node.to_json() for node in library.tree],
        }

    @app.get("/api/playlists/{playlist_id}")
    def get_playlist(playlist_id: int) -> dict:
        library = require_library()
        if playlist_id not in library.playlists:
            raise HTTPException(status_code=404, detail="Unknown playlist")
        node = library.playlists[playlist_id]
        return {
            "id": node.id,
            "name": node.name,
            "tracks": [track.to_json() for track in library.playlist_tracks(playlist_id)],
        }

    @app.post("/api/play")
    def play(request: PlayRequest) -> dict:
        library = require_library()
        if request.playlist_id not in library.playlists:
            raise HTTPException(status_code=404, detail="Unknown playlist")
        tracks = library.playlist_tracks(request.playlist_id)
        if not 0 <= request.index < len(tracks):
            raise HTTPException(status_code=400, detail="Track index out of range")
        engine.play(tracks, request.index, request.playlist_id, library.playlists[request.playlist_id].name)
        return engine.state()

    @app.post("/api/pause")
    def pause() -> dict:
        engine.pause()
        return engine.state()

    @app.post("/api/resume")
    def resume() -> dict:
        engine.resume()
        return engine.state()

    @app.post("/api/skip")
    def skip() -> dict:
        engine.skip()
        return engine.state()

    @app.post("/api/seek")
    def seek(request: SeekRequest) -> dict:
        engine.seek(request.position_s)
        return engine.state()

    @app.put("/api/master-bpm")
    def put_master_bpm(request: MasterBpmRequest) -> dict:
        engine.set_master_bpm(request.bpm)
        return engine.state()

    @app.post("/api/queue")
    def queue_add(request: QueueAddRequest) -> dict:
        library = require_library()
        track = library.tracks.get(request.track_id)
        if track is None:
            raise HTTPException(status_code=404, detail="Unknown track")
        playlist = library.playlists.get(request.playlist_id) if request.playlist_id is not None else None
        engine.queue_add(track, playlist.name if playlist else "", request.position)
        return engine.state()

    @app.post("/api/queue/{queue_id}/move")
    def queue_move(queue_id: int, request: QueueMoveRequest) -> dict:
        engine.queue_move(queue_id, request.position)
        return engine.state()

    @app.delete("/api/queue/{queue_id}")
    def queue_remove(queue_id: int) -> dict:
        engine.queue_remove(queue_id)
        return engine.state()

    @app.delete("/api/queue")
    def queue_clear() -> dict:
        engine.queue_clear()
        return engine.state()

    @app.post("/api/stop")
    def stop() -> dict:
        engine.stop()
        return engine.state()

    @app.put("/api/settings")
    def put_settings(update: SettingsUpdate) -> dict:
        engine.update_settings(**update.model_dump(exclude_none=True))
        return engine.state()

    @app.get("/api/state")
    def get_state() -> dict:
        return engine.state()

    @app.websocket("/api/state")
    async def state_socket(socket: WebSocket) -> None:
        await socket.accept()
        try:
            while True:
                try:
                    state = engine.state()
                    state["usb_connected"] = watcher.library is not None
                    message = json.dumps(state)
                except Exception:
                    log.exception("Could not build player state")
                else:
                    await socket.send_text(message)
                await asyncio.sleep(STATE_PUSH_SECONDS)
        except WebSocketDisconnect:
            return

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
