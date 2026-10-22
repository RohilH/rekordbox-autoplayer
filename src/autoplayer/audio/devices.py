"""The macOS default output device, read from CoreAudio.

PortAudio only learns about devices when it initializes, so it can't tell us when the system output
changes (AirPods connecting, say). CoreAudio can, without disturbing an open stream.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputDevice:
    id: int
    name: str


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


class _PropertyAddress(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


SYSTEM_OBJECT = 1
DEFAULT_OUTPUT_DEVICE = _fourcc("dOut")
OBJECT_NAME = _fourcc("lnam")
SCOPE_GLOBAL = _fourcc("glob")
ELEMENT_MAIN = 0
UTF8 = 0x08000100


def _load(name: str):
    path = ctypes.util.find_library(name)
    return ctypes.cdll.LoadLibrary(path) if path else None


_core_audio = _load("CoreAudio")
_core_foundation = _load("CoreFoundation")
if _core_audio is not None and _core_foundation is not None:
    _get_property = _core_audio.AudioObjectGetPropertyData
    _get_property.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_PropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    _get_property.restype = ctypes.c_int32
    _cf_string = _core_foundation.CFStringGetCString
    _cf_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    _cf_string.restype = ctypes.c_bool
    _cf_release = _core_foundation.CFRelease
    _cf_release.argtypes = [ctypes.c_void_p]


def _property(object_id: int, selector: int, value, size: int) -> bool:
    address = _PropertyAddress(selector, SCOPE_GLOBAL, ELEMENT_MAIN)
    length = ctypes.c_uint32(size)
    return _get_property(object_id, ctypes.byref(address), 0, None, ctypes.byref(length), ctypes.byref(value)) == 0


def default_output_device() -> OutputDevice | None:
    """The current system output, or None where CoreAudio isn't available."""
    if _core_audio is None or _core_foundation is None:
        return None
    device_id = ctypes.c_uint32(0)
    if not _property(SYSTEM_OBJECT, DEFAULT_OUTPUT_DEVICE, device_id, ctypes.sizeof(device_id)):
        return None
    name_ref = ctypes.c_void_p()
    name = ""
    if _property(device_id.value, OBJECT_NAME, name_ref, ctypes.sizeof(name_ref)) and name_ref.value:
        buffer = ctypes.create_string_buffer(512)
        if _cf_string(name_ref, buffer, len(buffer), UTF8):
            name = buffer.value.decode("utf-8", "replace")
        _cf_release(name_ref)
    return OutputDevice(device_id.value, name)
