from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML used by the dependency-light pilot."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

