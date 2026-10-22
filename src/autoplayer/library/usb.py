from __future__ import annotations

from pathlib import Path

from autoplayer.library.pdb import PDB_RELATIVE_PATH

VOLUMES = Path("/Volumes")


def find_usb_roots(volumes: Path = VOLUMES) -> list[Path]:
    if not volumes.is_dir():
        return []
    return sorted(v for v in volumes.iterdir() if (v / PDB_RELATIVE_PATH).is_file())
