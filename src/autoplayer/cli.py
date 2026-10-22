from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np

from autoplayer.audio.decode import SAMPLE_RATE
from autoplayer.audio.master import MasterBus
from autoplayer.audio.transition import MixSettings, Timeline, load_track
from autoplayer.library.anlz import load_analysis
from autoplayer.library.pdb import Library, PlaylistNode, load_library
from autoplayer.library.usb import find_usb_roots

RENDER_BLOCK_FRAMES = 1 << 16
TRANSITION_PADDING_SECONDS = 20


def _resolve_root(usb: str | None) -> Path:
    if usb:
        return Path(usb)
    roots = find_usb_roots()
    if not roots:
        sys.exit("No rekordbox USB found under /Volumes. Pass --usb PATH.")
    return roots[0]


def _print_tree(nodes: list[PlaylistNode], depth: int = 0) -> None:
    for node in nodes:
        label = f"{node.name}/" if node.is_folder else f"{node.name}  ({len(node.track_ids)} tracks, id {node.id})"
        print("  " * depth + label)
        _print_tree(node.children, depth + 1)


def cmd_list(args: argparse.Namespace) -> None:
    library = load_library(_resolve_root(args.usb))
    if args.beats is not None:
        track = library.tracks[args.beats]
        analysis = load_analysis(track.anlz_path)
        print(f"{track.title} — {track.artist}  {track.bpm:.2f} BPM  {track.audio_path}")
        print(f"{len(analysis.beats)} beats; first 8:")
        for beat in analysis.beats[:8]:
            print(f"  {beat.time_s:9.3f}s  bar pos {beat.bar_pos}  {beat.bpm:.2f} BPM")
        print("Cues:")
        for cue in analysis.cues:
            kind = "loop" if cue.is_loop else ("memory" if cue.hot_cue == 0 else f"hot {chr(64 + cue.hot_cue)}")
            print(f"  {cue.time_s:9.3f}s  {kind}")
        return
    if args.playlist:
        node = library.find_playlist(args.playlist)
        print(f"{node.name} (id {node.id})")
        for i, track in enumerate(library.playlist_tracks(node.id), 1):
            print(f"{i:3}. {track.title} — {track.artist}  [{track.bpm:.1f} BPM, {track.key}]  id {track.id}")
        return
    print(f"{library.root}: {len(library.tracks)} tracks")
    _print_tree(library.tree)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


def _render_range(timeline: Timeline, start: int, end: int) -> np.ndarray:
    master = MasterBus()
    blocks = [
        master.process(timeline.render(t, min(RENDER_BLOCK_FRAMES, end - t)))
        for t in range(start, end, RENDER_BLOCK_FRAMES)
    ]
    return np.concatenate(blocks) if blocks else np.zeros((0, 2), dtype=np.float32)


def render_playlist(library: Library, playlist: str, start: int, count: int, settings: MixSettings, full: bool):
    node = library.find_playlist(playlist)
    tracks = library.playlist_tracks(node.id)[start : start + count]
    timeline = Timeline(settings)
    for offset, track in enumerate(tracks):
        placement = timeline.append(load_track(track, start + offset))
        kind = "start" if offset == 0 else ("beat-matched" if placement.beatmatched else settings.mode)
        if placement.cues:
            kind += f" on cues {placement.cues}"
        rate = float(placement.curve.rate_at(placement.out_start))
        print(
            f"{start + offset + 1:3}. {track.title}: {kind} at {placement.out_start / SAMPLE_RATE:7.1f}s, "
            f"tempo {100 * (rate - 1):+.1f}%{' (key lock)' if placement.key_lock_end is not None else ''}"
        )
    timeline.prepare()
    if full:
        return _render_range(timeline, 0, timeline.end)
    padding = TRANSITION_PADDING_SECONDS * SAMPLE_RATE
    gap = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
    pieces = []
    for prev, nxt in zip(timeline.placements, timeline.placements[1:]):
        pieces += [_render_range(timeline, max(0, nxt.out_start - padding), prev.out_end + padding), gap]
    return np.concatenate(pieces) if pieces else np.zeros((0, 2), dtype=np.float32)


def cmd_render(args: argparse.Namespace) -> None:
    library = load_library(_resolve_root(args.usb))
    settings = MixSettings(
        mode=args.mode, overlap_bars=args.overlap_bars, ease_bars=args.ease_bars, key_lock=not args.no_key_lock
    )
    audio = render_playlist(library, args.playlist, args.start - 1, args.count, settings, args.full)
    _write_wav(Path(args.out), audio)
    print(f"Wrote {len(audio) / SAMPLE_RATE:.0f}s to {args.out}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from autoplayer.audio.engine import Engine
    from autoplayer.server.app import LibraryWatcher, create_app

    engine = Engine(device=args.device)
    watcher = LibraryWatcher(Path(args.usb) if args.usb else None)
    print(f"Open http://localhost:{args.port}")
    uvicorn.run(create_app(engine, watcher), host=args.host, port=args.port, log_level="warning")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="autoplayer", description="Play rekordbox USB playlists hands-free")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the player and its web UI")
    p_serve.add_argument("--usb", help="USB root (default: first rekordbox USB under /Volumes)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--device", help="audio output device name or index (see `python -m sounddevice`)")
    p_serve.set_defaults(func=cmd_serve)

    p_list = sub.add_parser("list", help="print playlists, a playlist's tracks, or a track's beat grid")
    p_list.add_argument("--usb")
    p_list.add_argument("--playlist", help="playlist name or id")
    p_list.add_argument("--beats", type=int, metavar="TRACK_ID", help="dump a track's beat grid and cues")
    p_list.set_defaults(func=cmd_list)

    p_render = sub.add_parser("render", help="render transitions (or a full mix) to a WAV file")
    p_render.add_argument("--usb")
    p_render.add_argument("--playlist", required=True, help="playlist name or id")
    p_render.add_argument("--from", dest="start", type=int, default=1, help="1-based track number to start at")
    p_render.add_argument("--count", type=int, default=2, help="number of tracks")
    p_render.add_argument("--mode", choices=["off", "timed", "beatmatch"], default="beatmatch")
    p_render.add_argument("--overlap-bars", type=int, default=MixSettings.overlap_bars)
    p_render.add_argument("--ease-bars", type=int, default=MixSettings.ease_bars)
    p_render.add_argument("--no-key-lock", action="store_true", help="let pitch follow tempo (master tempo off)")
    p_render.add_argument("--full", action="store_true", help="render the whole mix, not just the transitions")
    p_render.add_argument("--out", default="mix.wav")
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
