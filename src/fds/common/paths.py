"""Locate machine-local assets independently of an installed or frozen source tree."""

import os
from pathlib import Path


def workspace_root(explicit=None):
    configured = explicit or os.environ.get("FDS_WORKSPACE")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not (path / "configs/dependencies.toml").is_file():
            raise ValueError(f"Invalid FDS workspace: {path}")
        return path
    for path in (Path.cwd(), *Path(__file__).resolve().parents):
        if (path / "configs/dependencies.toml").is_file() and (path / "src/fds").is_dir():
            return path
    raise ValueError("Run from the repository or set FDS_WORKSPACE to the asset workspace")
