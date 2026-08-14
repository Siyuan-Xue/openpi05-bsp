# Resumable LIBERO DataLoader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume a legacy or new LIBERO training checkpoint at the exact next shuffled micro-batch and persist validated loader cursor metadata for later resumes.

**Architecture:** Add a dependency-light cursor schema and a replayable PyTorch-compatible batch sampler. Load or derive the cursor before the first training batch, keep the existing Orbax item layout, and write cursor JSON through the existing assets callback.

**Tech Stack:** Python 3.11, PyTorch DataLoader, JAX/Flax, Orbax, pytest, Ruff.

## Global Constraints

- Work only on `fix/phase1-resumable-loader`, based on runtime commit `2c098404a3cce0c86f0b863dcd8d3aeb18a55d94`.
- Do not modify `main`, `refactor/pi05-libero-bsp-slim`, completed baseline checkpoints, or the legacy BSP checkpoint.
- Preserve phase-one seed 42, effective batch 256, micro-batch 64, four accumulation steps, EMA disabled, and milestones `0/1000/2000/5000/10000`.
- Keep existing Orbax `params`, `train_state`, and `assets` item names.
- Apply TDD: each production change follows a focused failing test with the expected failure observed.
- Refuse resume before GPU optimization when cursor identity or topology differs.
- If pinned-server parity with the legacy loader cannot be proven, do not deploy; restart BSP from step zero instead.

---

### Task 1: Define and validate the loader cursor

**Files:**
- Create: `src/openpi/training/loader_resume.py`
- Create: `src/openpi/training/loader_resume_test.py`

**Interfaces:**
- Produces: `LoaderIdentity`, `LoaderCursor`, `cursor_for_step()`, `load_cursor()`, and `save_cursor()`.
- Consumes: only the standard library and a path object with `read_text`/`write_text` behavior.

- [ ] **Step 1: Write failing schema and round-trip tests**

```python
def _identity(**changes):
    values = dict(
        repo_id="physical-intelligence/libero",
        revision="v2.0",
        dataset_length=273_465,
        dataset_fingerprint="de4a79e770bcac3f",
        bsp_cache_fingerprint="db8fe671f0e0ad33dcf2ef2e563c779c0f6c2cc4d91e314379d1c0bc64768213",
        action_horizon=16,
        action_keys=("actions",),
        seed=42,
        shuffle=True,
        global_micro_batch_size=64,
        local_batch_size=64,
        accumulation_steps=4,
        process_count=1,
        num_workers=2,
        drop_last=True,
        sampler_protocol="torch-random-sampler-v1",
    )
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q src/openpi/training/loader_resume_test.py
```

Expected: collection fails because `openpi.training.loader_resume` does not exist.

- [ ] **Step 3: Implement the immutable schema and canonical JSON**

```python
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


@dataclasses.dataclass(frozen=True)
class LoaderCursor:
    format_version: int
    completed_step: int
    consumed_batches: int
    identity: LoaderIdentity
```

`cursor_for_step(step, identity)` validates nonnegative integer steps and returns `consumed_batches=step * identity.accumulation_steps`. `save_cursor` writes sorted compact JSON; `load_cursor` returns `None` only for a missing file and rejects malformed or unsupported content.

- [ ] **Step 4: Add failing mismatch tests**

Parameterize replacements for `seed`, `dataset_fingerprint`, `bsp_cache_fingerprint`, `global_micro_batch_size`, `local_batch_size`, `accumulation_steps`, `process_count`, `num_workers`, and `sampler_protocol`. Assert `cursor.validate(expected_identity, expected_step)` raises `ValueError` containing the mismatched field. Add tests rejecting boolean/negative steps and a consumed count unequal to `step * accumulation_steps`.

- [ ] **Step 5: Implement strict validation and run GREEN**

Run:

```bash
pytest -q src/openpi/training/loader_resume_test.py
```

Expected: all cursor tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/openpi/training/loader_resume.py src/openpi/training/loader_resume_test.py
git commit -m "feat: define resumable loader cursor"
```

### Task 2: Reproduce and seek into the legacy shuffle stream

**Files:**
- Modify: `src/openpi/training/loader_resume.py`
- Modify: `src/openpi/training/loader_resume_test.py`
- Modify: `src/openpi/training/data_loader.py`
- Modify: `src/openpi/training/data_loader_test.py`

**Interfaces:**
- Consumes: `LoaderCursor.consumed_batches`, the pinned PyTorch `Generator`, dataset length, batch size, and worker persistence mode.
- Produces: `ReplayableRandomBatchSampler`, yielding `list[int]` batches from an absolute batch offset.

- [ ] **Step 1: Write a failing exact-index parity test**

Use an index-only dataset returning its integer index. Compare the first three epochs from the existing loader construction (`shuffle=True`, seed 42) against the replayable sampler for dataset sizes 10, 17, and 273, batch sizes 4 and 8, and `num_workers` 0 and 2.

```python
@pytest.mark.parametrize("dataset_size,batch_size", [(10, 4), (17, 4), (273, 8)])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_replayable_sampler_matches_legacy_shuffle(dataset_size, batch_size, num_workers):
    legacy = collect_legacy_indices(dataset_size, batch_size, num_workers, seed=42, epochs=3)
    replayable = collect_replayable_indices(dataset_size, batch_size, num_workers, seed=42, epochs=3)
    assert replayable == legacy
```

- [ ] **Step 2: Run the parity test and verify RED**

Run the single parameterized test. Expected: failure because `ReplayableRandomBatchSampler` is absent.

- [ ] **Step 3: Implement the sampler with pinned legacy semantics**

The sampler owns the sampling generator, reconstructs PyTorch's legacy base-seed and `randperm` consumption for `persistent_workers = num_workers > 0`, drops incomplete batches, advances full epochs using indices only, and skips the in-epoch prefix for `start_batch`. The DataLoader receives a separate worker-seeding generator so worker prefetch cannot advance the sampling cursor.

The sampler must expose:

```python
class ReplayableRandomBatchSampler(torch.utils.data.Sampler[list[int]]):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int,
        start_batch: int = 0,
        drop_last: bool = True,
        legacy_persistent_workers: bool,
    ): ...

    def __iter__(self) -> Iterator[list[int]]: ...
    def __len__(self) -> int: ...
```

- [ ] **Step 4: Write failing offset and boundary tests**

For offsets `0`, `1`, `batches_per_epoch - 1`, `batches_per_epoch`, and `2 * batches_per_epoch + 1`, assert that resumed indices equal the uninterrupted suffix. Assert offsets never decode skipped samples by using a dataset that records `__getitem__` calls.

- [ ] **Step 5: Integrate the batch sampler in the JAX LeRobot loader**

Add optional `start_batch: int = 0` to `create_data_loader`, `create_torch_data_loader`, and `TorchDataLoader`. Use the replayable sampler only for JAX random-access training with `shuffle=True`. Reject nonzero offsets for RLDS, PyTorch DDP, or `shuffle=False`.

- [ ] **Step 6: Run GREEN and existing loader tests**

```bash
pytest -q src/openpi/training/loader_resume_test.py src/openpi/training/data_loader_test.py
```

Expected: exact parity and all existing loader tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/openpi/training/loader_resume.py src/openpi/training/loader_resume_test.py src/openpi/training/data_loader.py src/openpi/training/data_loader_test.py
git commit -m "feat: seek the LIBERO shuffle stream"
```

### Task 3: Persist cursor metadata without changing Orbax items

**Files:**
- Modify: `src/openpi/training/checkpoints.py`
- Create: `src/openpi/training/checkpoints_test.py`
- Modify: `src/openpi/training/data_loader.py`

**Interfaces:**
- Consumes: `LoaderCursor`, checkpoint root, checkpoint step, and loader identity.
- Produces: `load_loader_cursor(checkpoint_dir, step)` and cursor writing in the existing assets callback.

- [ ] **Step 1: Write failing checkpoint compatibility tests**

Create temporary legacy and new checkpoint directory shapes. Verify legacy missing cursor returns `None`, new cursor round-trips at `<root>/<step>/assets/data_loader_cursor.json`, and malformed JSON raises without changing files.

- [ ] **Step 2: Run and verify RED**

Run `pytest -q src/openpi/training/checkpoints_test.py`. Expected: import or attribute failure for the missing helpers.

- [ ] **Step 3: Implement path and load helpers**

```python
def load_loader_cursor(checkpoint_dir: epath.Path | str, step: int) -> LoaderCursor | None:
    path = epath.Path(checkpoint_dir) / str(step) / "assets" / CURSOR_FILENAME
    return load_cursor(path)
```

Extend `save_state` with a required keyword-only `loader_cursor: LoaderCursor`. Its `save_assets` callback writes norm stats first and then cursor JSON into the assets root. Do not register a fourth Orbax item.

- [ ] **Step 4: Expose loader identity before iteration**

`DataLoaderImpl.resume_identity()` returns the immutable identity computed from the raw dataset before transformation. For BSP, use the validated `BspCacheManifest.fingerprint`; for baseline use the LeRobot HF fingerprint and episode-boundary hash. Fake/debug data receives a stable `fake:<length>` identity.

- [ ] **Step 5: Run GREEN and checkpoint tests**

```bash
pytest -q src/openpi/training/loader_resume_test.py src/openpi/training/checkpoints_test.py src/openpi/training/data_loader_test.py
```

- [ ] **Step 6: Commit Task 3**

```bash
git add src/openpi/training/checkpoints.py src/openpi/training/checkpoints_test.py src/openpi/training/data_loader.py
git commit -m "feat: persist loader resume metadata"
```

### Task 4: Resume before reading the first training batch

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/train_test.py`
- Modify: `src/openpi/training/train_planning.py`
- Modify: `src/openpi/training/train_planning_test.py`

**Interfaces:**
- Consumes: latest complete checkpoint step, optional stored cursor, current loader identity, and `GradientAccumulationPlan`.
- Produces: `plan_loader_resume(restored_step, accumulation_steps, stored_cursor)` and exact `start_batch` for loader construction.

- [ ] **Step 1: Write failing pure resume-planning tests**

Cover fresh step zero, legacy step 2,000 deriving 8,000 consumed batches, stored cursor validation, mismatched step rejection, and restored step beyond `num_train_steps` rejection.

- [ ] **Step 2: Verify RED, implement the pure planner, and verify GREEN**

Run the focused `train_planning_test.py` case before and after the minimal implementation.

- [ ] **Step 3: Write a failing training integration test**

Run a small deterministic fake-dataset training in two forms: uninterrupted to step 4, and step 2 checkpoint plus resume to step 4. Capture dataset indices and assert both sequences and final TrainState trees are equal. Assert step 2 and step 4 assets contain valid cursor files.

- [ ] **Step 4: Reorder training initialization**

Before creating the loader:

1. initialize the checkpoint manager;
2. obtain the latest complete step when resuming;
3. load optional cursor metadata;
4. derive the absolute loader offset;
5. construct and validate the loader at that offset;
6. only then create the iterator and request the first batch;
7. restore model and optimizer state;
8. train from `restored_step + 1`.

At step zero and every saved step, call `cursor_for_step(updated_state_step, data_loader.resume_identity())` and pass it to `save_state`.

- [ ] **Step 5: Add fail-closed integration tests**

Assert training aborts before `compute_microbatch_grad` when seed, dataset fingerprint, BSP fingerprint, batch geometry, worker count, or sampler protocol differs. Assert `--resume` with a legacy cursor logs `legacy-derived`, while a new checkpoint logs `checkpoint`.

- [ ] **Step 6: Run focused GREEN tests**

```bash
pytest -q scripts/train_test.py src/openpi/training/train_planning_test.py src/openpi/training/checkpoints_test.py src/openpi/training/data_loader_test.py src/openpi/training/loader_resume_test.py
```

- [ ] **Step 7: Commit Task 4**

```bash
git add scripts/train.py scripts/train_test.py src/openpi/training/train_planning.py src/openpi/training/train_planning_test.py
git commit -m "fix: resume at the exact next LIBERO batch"
```

### Task 5: Document the runtime transition and recovery contract

**Files:**
- Modify: `docs/pi05_libero_bsp_phase1_server.md`
- Modify: `docs/pi05_libero_bsp_server_state.md`
- Modify: `scripts/pi05_libero_bsp_phase1_server_test.py`

**Interfaces:**
- Documents: exact feature branch/SHA gate, legacy versus stored cursor log markers, checkpoint validation, and manifest code transition.

- [ ] **Step 1: Write failing runbook contract assertions**

Require the runbook to contain `fix/phase1-resumable-loader`, `phase1-runtime-2c09840`, `--resume`, `data_loader_cursor.json`, `legacy-derived`, `checkpoint`, and a two-SHA transition record. Require every shell block to remain non-destructive and syntactically valid.

- [ ] **Step 2: Verify RED and update the runbook**

Document read-only post-image checks, exact commit checkout after server tests, a dry loader-resume gate, the formal command with all original short10k parameters plus `--resume`, and rollback to step-zero restart if parity fails.

- [ ] **Step 3: Run GREEN and documentation contracts**

```bash
pytest -q scripts/pi05_libero_bsp_phase1_server_test.py scripts/libero_revision_contract_test.py
```

- [ ] **Step 4: Commit Task 5**

```bash
git add docs/pi05_libero_bsp_phase1_server.md docs/pi05_libero_bsp_server_state.md scripts/pi05_libero_bsp_phase1_server_test.py
git commit -m "docs: add exact BSP resume procedure"
```

### Task 6: Local and server verification, push, and controlled deployment

**Files:**
- Verify only; no further production edits unless a failing test starts a new TDD cycle.

**Interfaces:**
- Produces: final feature commit SHA and server evidence for exact sampler parity and checkpoint resume.

- [ ] **Step 1: Run local static and dependency-light gates**

```bash
ruff check scripts/train.py src/openpi/training/loader_resume.py src/openpi/training/data_loader.py src/openpi/training/checkpoints.py
ruff format --check scripts/train.py src/openpi/training/loader_resume.py src/openpi/training/data_loader.py src/openpi/training/checkpoints.py
python -m unittest -q scripts.libero_revision_contract_test scripts.pi05_libero_bsp_phase1_server_test
git diff --check phase1-runtime-2c09840...HEAD
```

- [ ] **Step 2: Push only the feature branch**

```bash
git push --set-upstream origin fix/phase1-resumable-loader
```

Do not push or merge `main` or the slimming branch.

- [ ] **Step 3: Wait for image submission and inspect the server read-only**

Confirm instance state `Running`, `/root/openpi-bsp-work` exists, OSS is mounted at `/mnt/data`, server HEAD remains `2c098404...`, no GPU process is active, and list complete checkpoint steps with no Orbax temporary directories.

- [ ] **Step 4: Fetch and test without changing the active checkout**

Fetch the feature branch with HTTP/1.1 fallback if needed. Use a separate server worktree for tests. Run the focused pytest set in the existing OpenPI Python 3.11 environment, including exact legacy loader parity for the pinned PyTorch version and phase-one `num_workers` value.

- [ ] **Step 5: Run a non-mutating dry resume gate**

Load the latest legacy checkpoint, compute its expected next micro-batch index hash, and compare it to an uninterrupted seed-42 reference stream. Do not start GPU optimization and do not write into the formal checkpoint directory.

- [ ] **Step 6: Deploy the exact tested commit**

Only after all server gates pass, point the runtime checkout at the tested feature commit. Record the old SHA, new SHA, restored step, latest checkpoint hash, cursor source, dataset identity, BSP fingerprint, and command line in a new run-transition manifest outside the checkpoint directory.

- [ ] **Step 7: Resume exactly one BSP process**

Use the original config and experiment name, seed 42, effective batch 256, micro-batch 64, 10,000 total optimizer steps, EMA `None`, save interval 1,000, original assets/data/sidecar/checkpoint paths, W&B offline, XLA fraction 0.95, and `--resume`. Confirm the process survives initialization, reports the correct restored and next steps, and creates no new step-zero checkpoint.

- [ ] **Step 8: Establish post-resume monitoring**

Monitor the one GPU process at the existing scheduled times. At the next permanent checkpoint, verify `data_loader_cursor.json`, validate its consumed count and identity, and prove a second dry resume selects cursor source `checkpoint` without launching a duplicate training process.
