# Task 1 report — BSP math and versioned sidecar cache primitives

## Implementation

Added `openpi.training.bsp`, a standalone, MIT-attributed minimum port of the
relevant FITPACK chunking and knot-projection behavior from the read-only
`bspline-policy` author implementation. It does not import that repository at
runtime.

- `BspSettings` is a frozen protocol object that enforces the binding values:
  cubic degree 3, chunk size 10, 16 target rows, seven action controls plus
  one knot channel, maximum fit error 0.002, smoothing `1e-12`, stride 1,
  `relative_knots=False`, and eight decoded actions.
- `build_episode_targets` validates `[frames, 7]` finite actions, fits one
  full episode through FITPACK's adaptive `generate_knots` /
  `make_lsq_spline`, rejects tolerance failures, chunks and pads to 16 rows,
  and stores targets as controls-first `[segments, 16, 8]` float32 tensors.
  A compact uint32 mapping selects each frame's nearest future segment.
- `decode_actions` projects only descending knots with the author policy's
  `1e-6` rule, consumes all 16 knots plus the first 12 controls, evaluates
  exactly eight frame-index action points, and sets `extrapolate=False`.
- `BspCacheManifest` derives a stable SHA-256 fingerprint from canonical JSON
  source metadata and the fixed protocol. It validates its own stored
  contents/version. `write_sidecar_cache` uses `filelock`, a sibling temporary
  `.npz`, flush/fsync, and `os.replace`; `load_sidecar_cache` validates the
  manifest, expected fingerprint, array dtype/shape/finite values, and index
  bounds before returning a cache.

## Files

- `src/openpi/training/bsp.py` — BSP math, projection/decode, manifest, and
  atomic sidecar primitives.
- `src/openpi/training/bsp_test.py` — focused behavior tests.
- `.superpowers/sdd/pi05-libero-bsp-sdd-plan/task-1-report.md` — this report.

## TDD evidence

The tests in `bsp_test.py` were added before `bsp.py`. The pre-implementation
RED import command was:

```text
$ PYTHONPATH=src python3 -c 'import openpi.training.bsp'
ModuleNotFoundError: No module named 'openpi.training.bsp'
```

The test contracts cover immutable fixed settings; fitted target shape and
controls-first layout; twelve effective controls; nearest-future mapping;
end-row padding; code-faithful projection; invalid decode inputs; malformed,
short, and nonfinite episodes; deterministic fingerprints; round-trip cache
load; and stale-cache rejection.

## Commands and output

```text
$ git diff --check
$ python3 -m py_compile src/openpi/training/bsp.py src/openpi/training/bsp_test.py
$ python3 -m compileall -q src/openpi/training/bsp.py src/openpi/training/bsp_test.py
exit 0 for all static checks; no output
```

The local test command cannot start because the coding-only machine has no
pytest, NumPy, or SciPy, and installing/syncing dependencies is prohibited:

```text
$ PYTHONPATH=src python3 -m pytest src/openpi/training/bsp_test.py
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest

$ PYTHONPATH=src python3 -c 'import openpi.training.bsp'
ModuleNotFoundError: No module named 'numpy'
```

Required server GREEN gate (with project dependencies, including SciPy 1.15.3,
available):

```text
uv run pytest src/openpi/training/bsp_test.py
```

No NumPy/SciPy test result is claimed locally.

## Self-review

- Confirmed no runtime import from the sibling author repository.
- Confirmed the target order is controls `0:7`, knot `7`, with 16 rows and 12
  decoder controls.
- Confirmed mapping uses `searchsorted(..., side="left")` so exact segment
  starts are selected and late frames use the last segment.
- Confirmed cache archives use `allow_pickle=False`, validate cached data on
  load, and cannot be reused after a source/protocol fingerprint change.
- Confirmed syntax compilation, whitespace check, and 120-column scan locally.

## Concerns / server follow-up

FITPACK's candidate-knot behavior and floating-point reconstruction tolerance
need the required server GREEN run. This local host deliberately lacks the
NumPy/SciPy/pytest runtime, so numerical fitting, SciPy decode, file-lock
round-trip, and pytest assertions were not executed here.

## Fix round 1 — review findings

### Changed behavior

- Preserved the applicable upstream MIT copyright and permission notice in
  `src/openpi/training/bsp.py`, with a stable repository URL, pinned source
  revision `61ed5f42fced971d50a89b46417493790876ccd1`, and source-file path.
- Moved the sidecar existence check into `load_sidecar_cache`'s shared lock.
  A first reader now waits for a concurrent writer to publish atomically,
  checks the file only after acquiring that lock, then validates/loads it.

### Test-first evidence and changed files

Added `test_cache_reader_waits_for_writer_publication_under_the_shared_lock`
to `src/openpi/training/bsp_test.py` before the implementation change. Its
lock context publishes a valid cache while the reader waits: the old ordering
raises the missing-cache error before entering that context; the fixed
ordering returns the published targets and mapping.

Changed files:

- `src/openpi/training/bsp.py`
- `src/openpi/training/bsp_test.py`
- `.superpowers/sdd/pi05-libero-bsp-sdd-plan/task-1-report.md`

### Commands and output

```text
$ python3 -m py_compile src/openpi/training/bsp_test.py
exit 0; no output

$ PYTHONPATH=src python3 -m pytest src/openpi/training/bsp_test.py
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

The focused test was therefore written against the prior (RED) ordering but
cannot be executed locally: pytest, NumPy, and SciPy are intentionally absent
and dependency installation is prohibited. Required server GREEN gate remains:

```text
uv run pytest src/openpi/training/bsp_test.py
```

Post-change static commands and output:

```text
$ python3 -m py_compile src/openpi/training/bsp.py src/openpi/training/bsp_test.py
$ python3 -m compileall -q src/openpi/training/bsp.py src/openpi/training/bsp_test.py
$ awk 'length($0) > 120 { print FILENAME ":" FNR ":" length($0); exit 1 }' \
    src/openpi/training/bsp.py src/openpi/training/bsp_test.py
exit 0 for all; no output
```

No deferred minor test-coverage scope was changed.
