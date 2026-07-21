"""Persist orbit candidates as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _next_index(directory: Path) -> int:
    existing = sorted(directory.glob("orbit_*.json"))
    if not existing:
        return 1
    nums = []
    for path in existing:
        try:
            nums.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return (max(nums) + 1) if nums else 1


def save_candidate(
    candidate: dict[str, Any],
    directory: str | Path = "orbit_library",
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    idx = _next_index(directory)
    path = directory / f"orbit_{idx:03d}.json"
    # Convert numpy types
    payload = _to_jsonable(candidate)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_candidate(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_candidates(directory: str | Path = "orbit_library") -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("orbit_*.json"))


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return float(obj) if hasattr(obj, "__float__") else str(obj)
