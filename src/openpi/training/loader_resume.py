"""Validated metadata for resuming a shuffled training data stream."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CURSOR_FORMAT_VERSION = 1
CURSOR_FILENAME = "data_loader_cursor.json"
SAMPLER_PROTOCOL = "torch-random-sampler-v1"


@dataclasses.dataclass(frozen=True)
class LoaderIdentity:
    repo_id: str
    revision: str | None
    dataset_length: int
    dataset_fingerprint: str
    bsp_cache_fingerprint: str | None
    action_horizon: int
    action_keys: tuple[str, ...]
    seed: int
    shuffle: bool
    global_micro_batch_size: int
    local_batch_size: int
    accumulation_steps: int
    process_count: int
    num_workers: int
    drop_last: bool
    sampler_protocol: str = SAMPLER_PROTOCOL

    def validate(self) -> None:
        for name in (
            "dataset_length",
            "action_horizon",
            "global_micro_batch_size",
            "local_batch_size",
            "accumulation_steps",
            "process_count",
        ):
            _require_positive_integer(name, getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError(f"num_workers must be a nonnegative integer, got {self.num_workers!r}")
        for name in ("repo_id", "dataset_fingerprint", "sampler_protocol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise ValueError("revision must be None or a non-empty string")
        if self.bsp_cache_fingerprint is not None and (
            not isinstance(self.bsp_cache_fingerprint, str) or not self.bsp_cache_fingerprint
        ):
            raise ValueError("bsp_cache_fingerprint must be None or a non-empty string")
        if not self.action_keys or not all(isinstance(key, str) and key for key in self.action_keys):
            raise ValueError("action_keys must contain non-empty strings")
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if not isinstance(self.drop_last, bool):
            raise ValueError("drop_last must be a boolean")
        if self.global_micro_batch_size != self.local_batch_size * self.process_count:
            raise ValueError(
                "global_micro_batch_size must equal local_batch_size * process_count"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LoaderIdentity:
        fields = {field.name for field in dataclasses.fields(cls)}
        if set(value) != fields:
            raise ValueError("Loader identity fields do not match cursor format")
        parsed = dict(value)
        action_keys = parsed.get("action_keys")
        if not isinstance(action_keys, list) or not all(isinstance(key, str) for key in action_keys):
            raise ValueError("Loader identity action_keys must be a list of strings")
        parsed["action_keys"] = tuple(action_keys)
        identity = cls(**parsed)
        identity.validate()
        return identity


@dataclasses.dataclass(frozen=True)
class LoaderCursor:
    format_version: int
    completed_step: int
    consumed_batches: int
    identity: LoaderIdentity

    def _validate_internal(self) -> None:
        if self.format_version != CURSOR_FORMAT_VERSION:
            raise ValueError(
                f"format_version={self.format_version!r} is unsupported; expected {CURSOR_FORMAT_VERSION}"
            )
        _require_nonnegative_integer("completed_step", self.completed_step)
        _require_nonnegative_integer("consumed_batches", self.consumed_batches)
        self.identity.validate()
        expected_batches = self.completed_step * self.identity.accumulation_steps
        if self.consumed_batches != expected_batches:
            raise ValueError(
                f"consumed_batches={self.consumed_batches} does not match "
                f"completed_step * accumulation_steps={expected_batches}"
            )

    def validate(self, expected_identity: LoaderIdentity, *, expected_step: int) -> None:
        self._validate_internal()
        expected_identity.validate()
        _require_nonnegative_integer("expected_step", expected_step)
        if self.completed_step != expected_step:
            raise ValueError(
                f"completed_step={self.completed_step} does not match expected_step={expected_step}"
            )
        for field in dataclasses.fields(LoaderIdentity):
            stored = getattr(self.identity, field.name)
            requested = getattr(expected_identity, field.name)
            if stored != requested:
                raise ValueError(
                    f"Loader cursor {field.name} mismatch: stored={stored!r}, requested={requested!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LoaderCursor:
        if set(value) != {"format_version", "completed_step", "consumed_batches", "identity"}:
            raise ValueError("Loader cursor fields do not match cursor format")
        identity = value["identity"]
        if not isinstance(identity, Mapping):
            raise ValueError("Loader cursor identity must be an object")
        cursor = cls(
            format_version=value["format_version"],
            completed_step=value["completed_step"],
            consumed_batches=value["consumed_batches"],
            identity=LoaderIdentity.from_dict(identity),
        )
        cursor._validate_internal()
        return cursor


def cursor_for_step(completed_step: int, identity: LoaderIdentity) -> LoaderCursor:
    _require_nonnegative_integer("completed_step", completed_step)
    identity.validate()
    cursor = LoaderCursor(
        format_version=CURSOR_FORMAT_VERSION,
        completed_step=completed_step,
        consumed_batches=completed_step * identity.accumulation_steps,
        identity=identity,
    )
    cursor.validate(identity, expected_step=completed_step)
    return cursor


def save_cursor(path: os.PathLike[str] | str, cursor: LoaderCursor) -> None:
    cursor._validate_internal()
    Path(path).write_text(
        json.dumps(cursor.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_cursor(path: os.PathLike[str] | str) -> LoaderCursor | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Loader cursor at {path} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("Loader cursor must be a JSON object")
    return LoaderCursor.from_dict(value)


def _require_nonnegative_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")


def _require_positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
