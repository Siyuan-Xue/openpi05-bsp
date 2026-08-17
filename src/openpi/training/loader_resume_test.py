from __future__ import annotations

import dataclasses

import pytest

from openpi.training import loader_resume


def _identity(**changes) -> loader_resume.LoaderIdentity:
    values = {
        "repo_id": "physical-intelligence/libero",
        "revision": "v2.0",
        "dataset_length": 273_465,
        "dataset_fingerprint": "de4a79e770bcac3f",
        "bsp_cache_fingerprint": "db8fe671f0e0ad33dcf2ef2e563c779c0f6c2cc4d91e314379d1c0bc64768213",
        "action_horizon": 16,
        "action_keys": ("actions",),
        "seed": 42,
        "shuffle": True,
        "global_micro_batch_size": 64,
        "local_batch_size": 64,
        "accumulation_steps": 4,
        "process_count": 1,
        "num_workers": 2,
        "drop_last": True,
        "sampler_protocol": "torch-random-sampler-v1",
    }
    values.update(changes)
    return loader_resume.LoaderIdentity(**values)


def test_cursor_round_trip_preserves_phase_one_identity(tmp_path):
    cursor = loader_resume.cursor_for_step(2_000, _identity())
    path = tmp_path / "data_loader_cursor.json"

    loader_resume.save_cursor(path, cursor)

    assert loader_resume.load_cursor(path) == cursor
    assert cursor.consumed_batches == 8_000


def test_missing_legacy_cursor_returns_none(tmp_path):
    assert loader_resume.load_cursor(tmp_path / "missing.json") is None


def test_invalid_json_cursor_fails_closed(tmp_path):
    path = tmp_path / "data_loader_cursor.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        loader_resume.load_cursor(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seed", 43),
        ("dataset_fingerprint", "different-dataset"),
        ("bsp_cache_fingerprint", "different-sidecar"),
        ("global_micro_batch_size", 32),
        ("local_batch_size", 32),
        ("accumulation_steps", 8),
        ("process_count", 2),
        ("num_workers", 0),
        ("sampler_protocol", "unknown-sampler"),
    ],
)
def test_cursor_rejects_resume_identity_mismatches(field, replacement):
    stored = loader_resume.cursor_for_step(2_000, _identity())
    requested = dataclasses.replace(_identity(), **{field: replacement})

    with pytest.raises(ValueError, match=field):
        stored.validate(requested, expected_step=2_000)


@pytest.mark.parametrize("step", [-1, True, 1.5])
def test_cursor_rejects_invalid_completed_steps(step):
    with pytest.raises(ValueError, match="completed_step"):
        loader_resume.cursor_for_step(step, _identity())


def test_cursor_rejects_consumed_batch_count_that_does_not_match_step():
    malformed = dataclasses.replace(
        loader_resume.cursor_for_step(2_000, _identity()),
        consumed_batches=7_999,
    )

    with pytest.raises(ValueError, match="consumed_batches"):
        malformed.validate(_identity(), expected_step=2_000)


def test_save_cursor_rejects_internally_inconsistent_cursor(tmp_path):
    malformed = dataclasses.replace(
        loader_resume.cursor_for_step(2_000, _identity()),
        consumed_batches=7_999,
    )

    with pytest.raises(ValueError, match="consumed_batches"):
        loader_resume.save_cursor(tmp_path / "data_loader_cursor.json", malformed)


@pytest.mark.parametrize("action_keys", ["actions", ["actions"]])
def test_identity_rejects_non_tuple_action_keys(action_keys):
    with pytest.raises(ValueError, match="action_keys"):
        _identity(action_keys=action_keys).validate()


def test_identity_rejects_non_exact_string_action_key():
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="action_keys"):
        _identity(action_keys=(StringSubclass("actions"),)).validate()


def test_cursor_rejects_unsupported_format_version():
    malformed = dataclasses.replace(
        loader_resume.cursor_for_step(2_000, _identity()),
        format_version=2,
    )

    with pytest.raises(ValueError, match="format_version"):
        malformed.validate(_identity(), expected_step=2_000)


@pytest.mark.parametrize("format_version", [True, 1.0])
def test_cursor_rejects_non_integer_format_version(format_version):
    malformed = dataclasses.replace(
        loader_resume.cursor_for_step(2_000, _identity()),
        format_version=format_version,
    )

    with pytest.raises(ValueError, match="format_version"):
        malformed.validate(_identity(), expected_step=2_000)
