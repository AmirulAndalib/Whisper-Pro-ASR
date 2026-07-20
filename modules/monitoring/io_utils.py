"""Shared monitoring I/O helpers."""

import json
from typing import Any, Optional


def load_json_list_file(path: str) -> Optional[list[Any]]:
    """Load a JSON list from disk, returning None when missing or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, list):
        return None
    return loaded
