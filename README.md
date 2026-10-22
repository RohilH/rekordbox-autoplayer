# rekordbox autoplayer

Plays the playlists on a rekordbox USB export start to finish, with beat-matched crossfades.

```
uv sync
uv run autoplayer serve        # then open http://localhost:8765
```

The player finds the first mounted volume with `PIONEER/rekordbox/export.pdb`; pass `--usb /Volumes/NAME` to pick one. Audio runs at 48 kHz, the native rate of Mac outputs, and follows the macOS output device, switching within a second when it changes (AirPods connecting, say); pass `--device` to pin one instead (`uv run python -m sounddevice` lists them).

## How transitions work

- **Beat-matched** (default): mixed the way you'd do it on a mixer, over the last 16 bars of the outgoing track. The next track enters with its low end killed at 50% volume and comes up to full over 8 bars; at bar 8 the lows swap (incoming bass in, outgoing bass out); over the last 8 bars the outgoing track eases down to 80% and is cut at bar 16. The incoming track plays at the outgoing tempo with its pitch held (master tempo, on by default), then eases back to its own tempo over 8 bars. Kick attacks are aligned rather than raw grid lines, using the onsets measured across the actual overlap. Tracks more than 8% apart in tempo (after trying half and double time) fall back to a timed crossfade.
  - **Hot cues drive it when you've set them.** The transition starts with the outgoing track's E lined up on the incoming track's A, the bass swap falls on F (or the incoming track's B when there's no F), and the outgoing track is cut at G. Any subset works, so a track cued only A–C still enters at A over the default outgoing phrase. Cues rekordbox generated automatically (`CUE(Auto)`) and cue sets that aren't in order are ignored.
- **Crossfade**: an 8-second equal-power fade with no tempo change; the next track starts at its A cue if it has one.
- **Gapless**: the next track starts where the last one goes silent.

Settings changes apply to transitions that haven't been planned yet; the next transition is planned as soon as the current track starts.

Set a master BPM with − and + by the BPM readout (hold to repeat, or use the + and − keys), or click the number and type one, as fast or slow as you like (40–300). The readout changes as you click and the audio follows about a fifth of a second later, pitch held when master tempo is on; every following track then plays at that BPM instead of easing back to its own. Reset hands tempo back to the tracks.

**Browsing and search.** The sidebar starts with **All tracks**, every track on the USB, followed by your **Playlists**. Every track list has a search box (press **/** to jump to it, **Esc** to clear): words match title, artist or album, a key such as `8A` or `Am` also matches tracks in that key, and a number such as `128` or a range such as `120-128` matches BPM. Every word has to match, so `remix 4A 126-128` finds 4A remixes around 126–128 BPM. Starting playback from a search plays the results, in the order shown.

**Sorting.** Click the **#**, **Title**, **Artist**, **BPM** or **Key** header to sort a list, and again to reverse it. Keys sort around the Camelot wheel (1A, 1B, 2A, 2B … 12A, 12B), with tracks that have no key last. The playlist then plays in the order shown; the # column keeps each track's place in the original playlist.

**Up next.** Hover a track and use **+** to queue it, or the arrow to play it next, from any playlist. The queue panel on the right (toggle it with *Up next* in the header) lists what's queued; drag items or use ↑/↓ to reorder, × to remove. Queued tracks are mixed in ahead of the playlist, with the same beat-matched, cue-driven transitions, and the playlist picks up where it left off once the queue is empty. The queue lives in the player, so it survives skips, seeks and starting another playlist. Changes apply to the next transition until it starts; within about a second of it starting, a new choice goes after that track instead.

Click the progress bar (or use ←/→ when it has focus) to jump within the playing track. Blue marks the transitions in and out, and hatching marks parts the mix skips. Landing inside a transition plays both tracks there, tempo-matched and at that moment's crossfade levels; landing in a skipped intro restarts the track alone from that point.

## Other commands

```
uv run autoplayer list                          # playlist tree
uv run autoplayer list --playlist "Bangers Only" # tracks in a playlist
uv run autoplayer list --beats 1690             # a track's beat grid and cues
uv run autoplayer render --playlist "Bangers Only" --from 1 --count 4 --out mix.wav
```

`render` writes the transitions (20 s either side of each) to a WAV so you can check them by ear; add `--full` for the whole mix, or `--no-key-lock` to hear them with pitch following tempo.

Master tempo works like this: a varispeed playhead places every beat at the exact sample, and Rubber Band (via `pedalboard`) shifts the pitch back by the same amount over the stretch where a track runs off its native tempo. That stretch is rendered in the background when the transition is planned, minutes before it plays.

## Development

```
uv run pytest
```

`tests/test_pdb.py` runs against a real export: set `AUTOPLAYER_TEST_USB=/Volumes/NAME`, or copy `export.pdb` into `tests/fixtures/usb/PIONEER/rekordbox/`.

The PDB and ANLZ parsers in `src/autoplayer/library/_gen/` are generated from Deep Symmetry's [crate-digger](https://github.com/Deep-Symmetry/crate-digger) Kaitai schemas (in `ksy/`):

```
kaitai-struct-compiler -t python --python-package autoplayer.library._gen \
  --outdir src/autoplayer/library/_gen ksy/rekordbox_pdb.ksy ksy/rekordbox_anlz.ksy
```
