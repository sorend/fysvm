"""Run metadata helpers for reproducible experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_metadata(
    output_dir: str | Path,
    *,
    command: Iterable[str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Write command, environment, and lockfile metadata beside run outputs."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command) if command is not None else sys.argv,
        "config": config or {},
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "uv_lock_sha256": _file_sha256(Path("uv.lock")),
    }
    (path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "command.txt").write_text(" ".join(metadata["command"]) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
