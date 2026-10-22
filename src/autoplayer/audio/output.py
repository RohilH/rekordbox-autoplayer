from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import sounddevice as sd

from autoplayer.audio.decode import CHANNELS, SAMPLE_RATE
from autoplayer.audio.devices import default_output_device

log = logging.getLogger(__name__)

DEVICE_POLL_SECONDS = 1.0
CALLBACK_FRAMES = 1024


@dataclass
class PlaybackStats:
    """Counters for diagnosing dropouts: a dropout is the device asking for audio the player didn't have."""

    callbacks: int = 0
    device_underflows: int = 0
    dropouts: int = 0
    dropped_frames: int = 0
    max_callback_gap_ms: float = 0.0
    last_callback_frames: int = 0

    def to_json(self) -> dict:
        return asdict(self)


class BlockQueue:
    """Rendered audio waiting to be played, plus a count of frames the device has consumed."""

    def __init__(self) -> None:
        self._blocks: deque[np.ndarray] = deque()
        self._offset = 0
        self._lock = threading.Lock()
        self.queued = 0
        self.played = 0
        self.stats = PlaybackStats()
        self._flowing = False
        self._crossfade_tail = np.zeros((0, CHANNELS), dtype=np.float32)
        self._crossfade_done = 0

    def push(self, block: np.ndarray) -> None:
        with self._lock:
            tail, done = self._crossfade_tail, self._crossfade_done
            if done < tail.shape[0]:
                n = min(tail.shape[0] - done, block.shape[0])
                fade_in = ((np.arange(done, done + n) + 1) / (tail.shape[0] + 1)).astype(np.float32)[:, None]
                block = block.copy()
                block[:n] = block[:n] * fade_in + tail[done : done + n] * (1 - fade_in)
                self._crossfade_done = done + n
            self._blocks.append(block)
            self.queued += block.shape[0]

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros((n, CHANNELS), dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._blocks:
                block = self._blocks[0]
                take = min(n - filled, block.shape[0] - self._offset)
                out[filled : filled + take] = block[self._offset : self._offset + take]
                filled += take
                self._offset += take
                if self._offset == block.shape[0]:
                    self._blocks.popleft()
                    self._offset = 0
            self.queued -= filled
            self.played += filled
            if filled < n:
                self.stats.dropped_frames += n - filled if self._flowing or filled else 0
                if self._flowing:
                    self.stats.dropouts += 1
            self._flowing = filled == n
        return out

    def reset(self, fade_frames: int = 0, position: int = 0) -> None:
        """Drop queued audio and move the played counter to `position`, keeping a short fade-out to avoid a click."""
        with self._lock:
            tail = self._take_locked(fade_frames)
            self._blocks.clear()
            self._offset = 0
            self.queued = 0
            self._crossfade_tail, self._crossfade_done = np.zeros((0, CHANNELS), dtype=np.float32), 0
            if tail.shape[0]:
                ramp = np.linspace(1, 0, tail.shape[0], dtype=np.float32)[:, None]
                self._blocks.append(tail * ramp)
                self.queued = tail.shape[0]
            self.played = position - self.queued

    def splice(self, keep_frames: int, crossfade_frames: int) -> None:
        """Keep the next `keep_frames` of queued audio and replace the rest with whatever is pushed next,
        crossfading from the old audio into the new over `crossfade_frames` so the switch is seamless."""
        with self._lock:
            kept = self._take_locked(keep_frames + crossfade_frames)
            head, tail = kept[:keep_frames], kept[keep_frames:]
            self._blocks.clear()
            self._offset = 0
            self._blocks.append(head.copy())
            self.queued = head.shape[0]
            self._crossfade_tail, self._crossfade_done = tail.copy(), 0

    def _take_locked(self, n: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        offset = self._offset
        remaining = n
        for block in self._blocks:
            if remaining <= 0:
                break
            part = block[offset : offset + remaining]
            parts.append(part)
            remaining -= part.shape[0]
            offset = 0
        return np.concatenate(parts) if parts else np.zeros((0, CHANNELS), dtype=np.float32)


class AudioOutput:
    """Plays the queue on a device. With no device given it follows the system output, reopening the
    stream when that changes (AirPods connecting or disconnecting, say)."""

    def __init__(self, queue: BlockQueue, device: int | str | None = None) -> None:
        self.queue = queue
        self.paused = False
        self._device = device
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._system_device = default_output_device() if device is None else None
        self._last_callback: float | None = None
        self._stream = self._open()

    @property
    def device_name(self) -> str:
        with self._lock:
            return sd.query_devices(self._stream.device)["name"]

    def start(self) -> None:
        with self._lock:
            self._stream.start()
        if self._device is None:
            threading.Thread(target=self._follow_system_output, name="output-device", daemon=True).start()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            self._stream.stop()
            self._stream.close()

    def _open(self) -> sd.OutputStream:
        return sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=self._device,
            latency="high",
            blocksize=CALLBACK_FRAMES,
            callback=self._callback,
        )

    def _follow_system_output(self) -> None:
        while not self._closed.wait(DEVICE_POLL_SECONDS):
            current = default_output_device()
            if current is None or current == self._system_device:
                continue
            log.info("System output changed to %s; switching", current.name)
            try:
                self._reopen()
                self._system_device = current
            except Exception:
                log.exception("Could not switch to %s", current.name)

    def _reopen(self) -> None:
        with self._lock:
            if self._closed.is_set():
                return
            self._stream.stop()
            self._stream.close()
            # PortAudio reads the device list (and the system default) only when it initializes.
            sd._terminate()
            sd._initialize()
            self._stream = self._open()
            self._stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        now = time.perf_counter()
        stats = self.queue.stats
        stats.callbacks += 1
        stats.last_callback_frames = frames
        if status.output_underflow:
            stats.device_underflows += 1
        if self._last_callback is not None:
            stats.max_callback_gap_ms = max(stats.max_callback_gap_ms, (now - self._last_callback) * 1000)
        self._last_callback = now
        if self.paused:
            outdata.fill(0)
            return
        outdata[:] = self.queue.pull(frames)
