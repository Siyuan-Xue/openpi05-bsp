"""Shared serialization primitives for LIBERO evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import io
import json
from pathlib import Path
import tempfile
from typing import Any


def is_sha256(value: Any, *, require_string: bool = True) -> bool:
    return (
        (not require_string or isinstance(value, str))
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_text(path, csv_text(rows))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(text)
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True, allow_nan=False)
        output.write("\n")
