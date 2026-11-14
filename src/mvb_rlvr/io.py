"""Read UTF-8 JSON data and replace JSON or JSONL outputs atomically."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and missing parents, then return its path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    """Decode a UTF-8 JSON file and return its value."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON-serializable data via a temporary file beside the target."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(target.parent), delete=False, suffix=".tmp"
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(target)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Return JSON objects from nonblank lines; reject other JSON values."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            records.append(item)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write object rows through a temporary file and replace the target."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(target.parent), delete=False, suffix=".tmp"
        ) as handle:
            tmp_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        tmp_path.replace(target)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
