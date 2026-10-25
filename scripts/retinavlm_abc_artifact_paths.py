"""Resolve paths recorded on the cluster into the verified local mirror."""

from __future__ import annotations

from pathlib import Path
import re


def local_artifact(path: str | Path, remote_root: Path | None,
                   local_root: Path | None) -> Path:
    source = Path(path)
    if remote_root is None and local_root is None:
        return source
    if remote_root is None or local_root is None:
        raise ValueError("Pass both remote and local artifact roots")
    try:
        relative = source.relative_to(remote_root)
    except ValueError as error:
        raise ValueError(f"Artifact path is outside the frozen cluster root: {source}") from error
    if ".." in relative.parts:
        raise ValueError(f"Artifact path escapes the frozen cluster root: {source}")
    if (len(relative.parts) == 3 and relative.parts[0] == "abc_training"
            and re.fullmatch(r"[ABC]_seed\d+", relative.parts[1])
            and re.fullmatch(r"checkpoint_step_\d{6}\.pt", relative.parts[2])):
        return local_root / relative.parts[0] / relative.parts[1] / "checkpoints" / relative.parts[2]
    return local_root / relative
