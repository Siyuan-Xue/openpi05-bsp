# Task 2 report — LeRobot BSP integration, preparation, and normalization assets

## Implementation

- Added `openpi.training.bsp_dataset` as the only BSP/LeRobot integration
  boundary. It validates the locked LeRobot `hf_dataset` and
  `episode_data_index` APIs, checks the supported LIBERO corpus metadata
  (1693 episodes, 273465 frames, 40 tasks, 10 fps), fingerprints the concrete
  HF table plus requested revision, builds a globally indexed sidecar from raw
  seven-dimensional full-episode actions, and wraps a standard LeRobot dataset.
- `BspLeRobotDataset` calls the standard dataset first, shallow-copies only the
  returned mapping, preserves observation/task/prompt values and object
  identities, then replaces only the configured action value via the
  precomputed global frame mapping. It contains no fitting path.
- Extended `DataConfig` and `LeRobotLiberoDataConfig` with explicit LeRobot
  root/revision and BSP enable/cache settings. The standard loader remains the
  baseline path. BSP mode rejects an unset cache path, constructs the same
  standard horizon-16 LeRobot dataset, validates and loads the sidecar, then
  applies prompt extraction and all existing transforms normally.
- Added `pi05_libero_baseline_h16` and the data-side
  `pi05_libero_bsp_h16`. Both use action horizon 16. Their normalization assets
  are separated as `libero_baseline_h16` and `libero_bsp_h16`.
- Added `scripts/prepare_libero_bsp.py` with explicit `download`, `build`, and
  `verify` modes. Every mode requires a persistent dataset root; build/verify
  additionally require a cache path. Build and verify refuse a missing local
  LeRobot metadata tree. Build is the only workflow that fits episodes.
- Fixed `scripts/compute_norm_stats.py` to write through `data_config.asset_id`
  rather than `repo_id`. It accepts exact persistent assets, BSP cache, and
  LeRobot dataset-root overrides. Its transform list intentionally stops before
  model transforms, so BSP statistics consume compact `[16, 8]` values and
  baseline statistics consume raw `[16, 7]` windows before either is padded to
  32 dimensions.

No runtime import from the sibling author repository was added, and
`scripts/train.py` was not modified.

## Files

- `src/openpi/training/bsp_dataset.py`
- `src/openpi/training/bsp_dataset_test.py`
- `src/openpi/training/data_loader.py`
- `src/openpi/training/data_loader_test.py`
- `src/openpi/training/config.py`
- `scripts/prepare_libero_bsp.py`
- `scripts/prepare_libero_bsp_test.py`
- `scripts/compute_norm_stats.py`
- `scripts/compute_norm_stats_test.py`
- `.superpowers/sdd/pi05-libero-bsp-sdd-plan/task-2-report.md`

## TDD evidence

The Task 2 tests were written and compiled before production changes. The
wrapper tests use a tiny real in-memory dataset double with real samples,
  episode boundaries, and an HF-like action table; they do not assert against
mocks. They cover action-only replacement by global mapping, preservation of
observation identity/task/prompt, source-sample immutability, full mapping
  coverage, table and episode-boundary fingerprint changes, episode target offsets, raw action
shape, and metadata rejection. Separate tests cover h16 asset/config isolation,
missing-cache refusal, explicit preparation paths/modes, and asset-ID output.

Pre-implementation dependency-free RED:

```text
$ python3 -m py_compile src/openpi/training/bsp_dataset_test.py \
    src/openpi/training/data_loader_test.py scripts/compute_norm_stats_test.py \
    scripts/prepare_libero_bsp_test.py
exit 0; no output

$ python3 - <<'PY'
# stdlib static contract probe for the production symbols required by the tests
...
PY
RED: Task 2 production contracts are absent
src/openpi/training/bsp_dataset.py: missing file
scripts/prepare_libero_bsp.py: missing file
```

The real pytest command cannot start on this coding-only host:

```text
$ PYTHONPATH=src python3 -m pytest src/openpi/training/bsp_dataset_test.py \
    scripts/compute_norm_stats_test.py scripts/prepare_libero_bsp_test.py \
    src/openpi/training/data_loader_test.py
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
exit 1
```

No local runtime GREEN is claimed. Required server gates are:

```text
uv run pytest src/openpi/training/bsp_test.py \
    src/openpi/training/bsp_dataset_test.py \
    src/openpi/training/data_loader_test.py \
    scripts/prepare_libero_bsp_test.py \
    scripts/compute_norm_stats_test.py
uv run ruff check src/openpi/training/bsp_dataset.py \
    src/openpi/training/bsp_dataset_test.py \
    src/openpi/training/data_loader.py src/openpi/training/data_loader_test.py \
    src/openpi/training/config.py scripts/prepare_libero_bsp.py \
    scripts/prepare_libero_bsp_test.py scripts/compute_norm_stats.py \
    scripts/compute_norm_stats_test.py
```

## Local verification

```text
$ python3 -m py_compile <all changed Python files>
$ python3 -m compileall -q <all changed Python files plus bsp.py>
$ git diff --check
exit 0 for all; no output

$ python3 - <<'PY'
# dependency-free AST contract: integration symbols exist and __getitem__ has
# no call to build_episode_targets
...
PY
STATIC CONTRACT PASS: action-only wrapper has no fit call; h16 assets/configs, boundary fingerprint, v2.1, CLI, and asset output present

$ python3 - <<'PY'
# dependency-free execution of the real BspLeRobotDataset with import stubs
...
PY
DEPENDENCY-FREE WRAPPER PASS: standard fields preserved by identity; mapped action replaced; source untouched
```

## Self-review

- Confirmed baseline creation takes the same code path unless `use_bsp=True`.
- Confirmed BSP mode rejects an unset path before constructing LeRobot and
  rejects missing, malformed, or stale caches through Task 1 cache validation.
- Confirmed the expected manifest is derived from the same standard dataset
  instance whose current-frame sample is wrapped.
- Confirmed the manifest covers the requested and metadata revisions, concrete
  HF table fingerprint, exact episode-boundary hash, and expected aggregate
  metadata; changed episode splits therefore invalidate a sidecar even when
  frame counts and table fingerprint remain unchanged.
- Confirmed the new baseline/BSP configs and preparation CLI consistently use
  LeRobot's valid `v2.1` dataset version rather than the invalid pseudo-version
  `main`; an explicit compatible version can still be supplied consistently.
- Confirmed episode fitting reads raw `hf_dataset[episode_start:episode_end]`
  seven-dimensional `actions`; it never reads normalized or horizon-windowed
  samples.
- Confirmed per-episode target indices are offset before global mappings are
  concatenated and the final mapping must cover every standard sample.
- Confirmed prompt extraction remains after the wrapper, so standard
  `task_index` semantics and `dataset_meta.tasks` remain unchanged.
- Confirmed normalization skips model transforms; compact BSP targets reach
  stats before `PadStatesAndActions` expands the last dimension to 32.
- Confirmed preparation and normalization locations are explicit/overridable,
  no repository-local cache default was introduced, and the training loader
  has no fitting fallback.
- Confirmed no Task 3 model/loss/microbatch implementation and no
  `scripts/train.py` changes are included.

## Independent review

The first review found three issues before commit: the invalid/mutable `main`
revision for pinned LeRobot, missing tiny-fixture metadata arguments in one
test, and omission of exact episode boundaries from the cache identity. The
fixes use valid `v2.1` consistently for the new baseline/BSP configs and CLI,
pass `TINY_METADATA` in the test, and include a canonical little-endian SHA-256
of the exact `from`/`to` arrays plus the loaded metadata revision. The focused
re-review confirmed all prior findings addressed and found no new Critical or
Important issues; its fresh `compileall` and `git diff --check` also passed.

## Concerns / server follow-up

- Runtime confirmation against pinned LeRobot revision
  `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` is required to validate its
  constructor/root behavior and concrete Arrow slicing/tensor conversions.
- The complete LIBERO build must verify FITPACK behavior for all 1693 episodes,
  the expected final target count, the table fingerprint's stability across
  preparation and training processes, and sidecar memory/I/O characteristics.
- NumPy/SciPy/JAX/Torch/LeRobot integration, tyro CLI parsing, and ruff are
  intentionally unrun locally because installing/syncing dependencies and
  downloading data/assets are prohibited.
