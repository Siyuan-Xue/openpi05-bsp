# Phase-One 0k/5k Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the fixed π0.5 + LIBERO Baseline/BSP phase-one protocol to save, retain, evaluate, and report checkpoints at 0k, 5k, 10k, 20k, and 30k optimizer steps.

**Architecture:** `TrainConfig` declares exact permanent checkpoint steps while the existing 1k save interval and 10k keep period remain intact. The JAX trainer writes a real step-0 Orbax checkpoint after base-weight loading and before any optimizer update, and resume treats step 0 as valid. The report layer accepts exactly ten runs, infers a consistent full/LoRA family from manifest config names, and produces five fixed milestone comparisons.

**Tech Stack:** Python 3.11, JAX/Flax NNX, Orbax Checkpoint 0.11.13, Tyro, unittest/pytest, OpenPI client report utilities.

## Global Constraints

- Fixed milestones are exactly `(0, 5_000, 10_000, 20_000, 30_000)`.
- Training still runs 30,000 optimizer steps with seed 42 and effective batch 256.
- Recovery checkpoints are still saved every 1,000 optimizer steps.
- Do not permanently retain 15k or 25k under the default phase-one configuration.
- Step 0 means `pi05_base` weights loaded and zero optimizer updates completed.
- Step 0 uses the experiment's own Baseline or BSP norm assets and inference protocol.
- A comparison may contain either ten full-finetune runs or ten LoRA runs, never a mixture.
- Each run still evaluates 2,000 episodes; ten runs total 20,000 episodes.
- Do not use official `pi05_libero` as a step-0 substitute.
- Do not change the BSP target representation, decoder, loss, deterministic seed derivation, or evaluation rollout protocol.
- Use test-first development: each production behavior must first be demonstrated by a failing test.

---

### Task 1: Declare and validate exact permanent checkpoint steps

**Files:**
- Modify: `src/openpi/training/config.py:483-575`
- Modify: `src/openpi/training/config.py:798-906`
- Modify: `src/openpi/training/data_loader_test.py:120-188`
- Modify: `src/openpi/training/train_planning.py:90-130`
- Modify: `src/openpi/training/train_planning_test.py:108-150`

**Interfaces:**
- Produces: `TrainConfig.permanent_checkpoint_steps: tuple[int, ...]`
- Produces: `train_planning.should_keep_checkpoint(step: int, *, permanent_steps: tuple[int, ...], keep_period: int | None) -> bool`
- Consumes: existing `TrainConfig.keep_period` as the periodic preservation rule

- [ ] **Step 1: Add failing phase-one configuration assertions**

Add this expectation to both full and LoRA phase-one config tests in `data_loader_test.py`:

```python
expected_milestones = (0, 5_000, 10_000, 20_000, 30_000)
assert config.permanent_checkpoint_steps == expected_milestones
assert config.keep_period == 10_000
```

Also include `permanent_checkpoint_steps` in the full/LoRA equality field list.

- [ ] **Step 2: Add failing exact-retention tests**

Add tests in `train_planning_test.py` that require:

```python
permanent = (0, 5_000, 10_000, 20_000, 30_000)
kept = [
    step
    for step in (0, 1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000)
    if should_keep_checkpoint(step, permanent_steps=permanent, keep_period=10_000)
]
assert kept == [0, 5_000, 10_000, 20_000, 30_000]
```

Add validation cases for negative steps, boolean steps, nonpositive `keep_period`, duplicate permanent steps, and unsorted permanent steps.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
pytest -q \
  src/openpi/training/data_loader_test.py::test_libero_h16_configs_use_the_same_jax_full_finetuning_recipe \
  src/openpi/training/data_loader_test.py::test_libero_h16_lora_configs_preserve_the_phase_one_recipe \
  src/openpi/training/train_planning_test.py
```

Expected: failures because `permanent_checkpoint_steps` and `should_keep_checkpoint` do not exist.

- [ ] **Step 4: Implement the configuration field and preservation predicate**

Add to `TrainConfig`:

```python
# Exact optimizer steps that must survive normal max-to-keep cleanup.
permanent_checkpoint_steps: tuple[int, ...] = ()
```

Extend `__post_init__` to reject booleans, negative values, duplicates, and nonascending tuples. Do not require values to be at most `num_train_steps`, because short pilot overrides intentionally use the same phase-one config.

Implement in `train_planning.py`:

```python
def should_keep_checkpoint(
    step: int,
    *,
    permanent_steps: tuple[int, ...],
    keep_period: int | None,
) -> bool:
    """Return whether Orbax must preserve a checkpoint beyond max_to_keep."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a nonnegative integer, got {step!r}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in permanent_steps):
        raise ValueError("permanent_steps must be unique nonnegative integers in ascending order")
    if tuple(sorted(permanent_steps)) != permanent_steps or len(set(permanent_steps)) != len(permanent_steps):
        raise ValueError("permanent_steps must be unique nonnegative integers in ascending order")
    if keep_period is not None:
        _require_positive_integer("keep_period", keep_period)
    return step in permanent_steps or (keep_period is not None and step % keep_period == 0)
```

Set the four phase-one configs to:

```python
permanent_checkpoint_steps=(0, 5_000, 10_000, 20_000, 30_000),
```

Keep `keep_period=10_000`; the exact predicate unions the two rules without preserving 15k/25k.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 3. Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/openpi/training/config.py \
  src/openpi/training/data_loader_test.py \
  src/openpi/training/train_planning.py \
  src/openpi/training/train_planning_test.py
git commit -m "feat: declare phase-one checkpoint milestones"
```

### Task 2: Save and resume a real step-0 checkpoint

**Files:**
- Modify: `src/openpi/training/checkpoints.py:20-62`
- Modify: `scripts/train.py:248-370`
- Modify: `scripts/train_test.py:55-90`

**Interfaces:**
- Consumes: `TrainConfig.permanent_checkpoint_steps`
- Consumes: `train_planning.should_keep_checkpoint(...)`
- Changes: `checkpoints.initialize_checkpoint_dir(..., permanent_checkpoint_steps: tuple[int, ...])`
- Produces: a normal Orbax `0/` checkpoint containing `params`, `train_state`, and experiment assets

- [ ] **Step 1: Add a failing fresh step-0 integration test**

Add a tiny `debug` training case in `scripts/train_test.py`:

```python
def test_train_saves_step_zero_before_any_optimizer_update(tmp_path):
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="step-zero",
        num_train_steps=0,
        permanent_checkpoint_steps=(0,),
        keep_period=None,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    )
    train.main(config)
    checkpoint = tmp_path / "checkpoint" / "debug" / "step-zero" / "0"
    assert (checkpoint / "params").is_dir()
    assert (checkpoint / "train_state").is_dir()
```

Because `num_train_steps=0`, the existence of this normal checkpoint structure proves it was written before any optimizer update. The resume test below verifies it is restorable.

- [ ] **Step 2: Add a failing step-0 resume integration test**

Construct the same config with `num_train_steps=0`, run it once, then replace with `resume=True, num_train_steps=2, save_interval=2`. Assert:

```python
assert (checkpoint_dir / "0" / "params").is_dir()
assert (checkpoint_dir / "2" / "params").is_dir()
assert not (checkpoint_dir / "1").exists()
```

The log/restore behavior must demonstrate that the second run resumed the existing step 0 rather than trying to save step 0 again.

- [ ] **Step 3: Run focused integration tests and verify RED**

Run:

```bash
pytest -q scripts/train_test.py -k 'step_zero or test_train'
```

Expected: the new tests fail because the trainer neither writes nor resumes step 0.

- [ ] **Step 4: Wire exact preservation into Orbax**

Extend `initialize_checkpoint_dir` with `permanent_checkpoint_steps`. When the tuple is nonempty, pass a `functools.partial` of `train_planning.should_keep_checkpoint` as `CheckpointManagerOptions.should_keep_fn`. Because `should_keep_fn` supersedes Orbax's `keep_period`, pass the union predicate and do not separately pass `keep_period` in that case. For configs without exact steps, preserve the existing `keep_period` option unchanged.

Change the empty-resume special case from:

```python
if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
```

to:

```python
if resuming and tuple(mngr.all_steps()) == ():
```

This makes step 0 a valid restore boundary.

- [ ] **Step 5: Save step 0 before optimizer compilation/update**

Pass `config.permanent_checkpoint_steps` into `initialize_checkpoint_dir`. After `init_train_state` completes:

```python
if resuming:
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
elif 0 in config.permanent_checkpoint_steps:
    if int(train_state.step) != 0:
        raise RuntimeError("Initial checkpoint requires train state step 0 before optimization.")
    _checkpoints.save_state(checkpoint_manager, train_state, data_loader, 0)
    checkpoint_manager.wait_until_finished()
```

Keep this block before the optimizer-step loop and before any `compute_microbatch_grad` call can execute.

- [ ] **Step 6: Run focused integration tests and verify GREEN**

Run the command from Step 3. Expected: all selected tests pass and both `0/params` and `2/params` exist in the resume test.

- [ ] **Step 7: Run checkpoint/planning regression tests**

```bash
pytest -q \
  src/openpi/training/train_planning_test.py \
  scripts/train_test.py
```

Expected: all tests pass; the existing non-step-zero debug flow remains unchanged.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/openpi/training/checkpoints.py scripts/train.py scripts/train_test.py
git commit -m "feat: save auditable step-zero checkpoints"
```

### Task 3: Expand the fixed comparison to ten full-or-LoRA runs

**Files:**
- Modify: `packages/openpi-client/src/openpi_client/libero_report.py:18-30`
- Modify: `packages/openpi-client/src/openpi_client/libero_report.py:175-315`
- Modify: `packages/openpi-client/src/openpi_client/libero_report.py:715-890`
- Modify: `packages/openpi-client/src/openpi_client/libero_report.py:994-1110`
- Modify: `packages/openpi-client/src/openpi_client/libero_report_test.py:1-320`
- Modify: `scripts/compare_libero_phase1.py:10-65`
- Modify: `scripts/compare_libero_phase1_test.py:1-40`

**Interfaces:**
- Changes: `libero_report.MILESTONES = (0, 5000, 10000, 20000, 30000)`
- Produces: `_training_family(manifest: Mapping[str, Any]) -> str` returning `"full"` or `"lora"`
- Changes: `compare_phase_one(run_dirs, ...)` requires exactly ten run directories
- Adds: `comparison["protocol"]["training_family"]`

- [ ] **Step 1: Change report fixtures and add failing 20,000-rollout expectations**

In `libero_report_test.py` set:

```python
_STEPS = (0, 5000, 10000, 20000, 30000)
```

Update the full comparison test to build ten deliberately shuffled runs and assert:

```python
assert comparison["protocol"]["milestones"] == list(_STEPS)
assert comparison["protocol"]["total_episodes"] == 20_000
assert comparison["protocol"]["training_family"] == "full"
assert len(task_rows) == 200
assert len(suite_rows) == 20
assert len(learning_rows) == 5
```

- [ ] **Step 2: Add failing full/LoRA family tests**

Make `_manifest(variant, step, *, family="full")` choose config names from:

```python
{
    ("baseline", "full"): "pi05_libero_baseline_h16",
    ("bsp", "full"): "pi05_libero_bsp_h16",
    ("baseline", "lora"): "pi05_libero_baseline_lora_h16",
    ("bsp", "lora"): "pi05_libero_bsp_lora_h16",
}
```

Add one test accepting all ten LoRA manifests and one test replacing a single LoRA manifest with a full manifest and expecting `ComparisonError` containing `training family`.

- [ ] **Step 3: Add failing CLI cardinality tests**

Update `compare_libero_phase1_test.py` to pass ten run paths and assert nine paths fail parsing. Also assert help/description text refers to ten runs and all five milestones.

- [ ] **Step 4: Run focused report tests and verify RED**

Run:

```bash
pytest -q \
  packages/openpi-client/src/openpi_client/libero_report_test.py \
  scripts/compare_libero_phase1_test.py
```

Expected: failures on old milestones, six-run cardinality, rejected LoRA names, and old episode totals.

- [ ] **Step 5: Implement five milestones and training-family validation**

Set the five milestones and derived `TOTAL_EPISODES`. Add the four-entry config mapping shown in Step 2 and a helper that rejects any config name outside the mapping. `_validate_manifest` must validate variant/config pairing. `classify_phase_one_manifests` must:

```python
required_run_count = 2 * len(MILESTONES)
if len(manifests) != required_run_count:
    raise ComparisonError(
        f"Phase-one comparison requires exactly {required_run_count} run manifests"
    )
families = {_training_family(manifest) for manifest in manifests}
if len(families) != 1:
    raise ComparisonError("All phase-one runs must use one training family")
```

Replace all hardcoded `six`/`6` validation messages with derived counts. Keep dictionary keys as `(variant, step)` so the remaining pairing/report code stays focused.

- [ ] **Step 6: Update generated report metadata and prose**

Add `training_family` to the protocol object. Update the Markdown report sentence to state that all five fixed checkpoints are compared and no best checkpoint is selected. The SVG x-axis must handle step 0 without extrapolation or division-by-zero; existing min/max scaling should be covered by the five-point test.

- [ ] **Step 7: Update the CLI to ten runs**

Set `nargs=10`, update help text and success output to `0k/5k/10k/20k/30k`, while keeping the three explicit diagnostics/output flags unchanged.

- [ ] **Step 8: Run focused report tests and verify GREEN**

Run the command from Step 4. Expected: all report and CLI tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add packages/openpi-client/src/openpi_client/libero_report.py \
  packages/openpi-client/src/openpi_client/libero_report_test.py \
  scripts/compare_libero_phase1.py \
  scripts/compare_libero_phase1_test.py
git commit -m "feat: compare five phase-one milestones"
```

### Task 4: Align the server runbook and contract tests

**Files:**
- Modify: `docs/pi05_libero_bsp_phase1_server.md:995-1420`
- Modify: `docs/pi05_libero_bsp_server_state.md:405-425`
- Modify: `docs/pi05_libero_bsp_server_state.md:765-785`
- Modify: `scripts/pi05_libero_bsp_phase1_server_test.py:80-205`

**Interfaces:**
- Consumes: the fixed milestone tuple and ten-run CLI from Tasks 1-3
- Produces: executable server instructions for step-0 validation, five retained steps, ten evaluations, and 20,000 episodes

- [ ] **Step 1: Add failing documentation contract expectations**

Update `pi05_libero_bsp_phase1_server_test.py` to require the runbook to contain all five milestone directory names, `20,000 episodes`, ten unique checkpoint paths, and ten comparison inputs. Remove assertions that require `12,000 episodes`, six checkpoints, or `--keep-period 10000` as the sole retention mechanism.

- [ ] **Step 2: Run the contract test and verify RED**

```bash
pytest -q scripts/pi05_libero_bsp_phase1_server_test.py
```

Expected: failures because the runbook still documents six runs and 12,000 episodes.

- [ ] **Step 3: Update formal training and step-0 gates in the runbook**

Document that both formal commands inherit `permanent_checkpoint_steps` from their selected full or LoRA config. Add post-start commands that require both `0/params` directories and validate that the checkpoint terminal component is `0`. Preserve `save_interval=1000`, seed 42, batch 256, and the chosen micro-batch/EMA decision.

Replace every phase-one evaluation reference to six checkpoints/12,000 episodes with ten checkpoints/20,000 episodes. List the terminal step directories as `0`, `5000`, `10000`, `20000`, and `30000` for each variant. Do not alter the unrelated section titled “六个外网端点”.

- [ ] **Step 4: Update comparison commands and acceptance text**

Provide ten run-directory arguments to `scripts/compare_libero_phase1.py`, ordered or shuffled without relying on path order. State that the comparator classifies manifests and rejects missing 0k/5k, mixed full/LoRA families, and any incomplete 20,000-rollout input set. Keep the six output artifact filenames unchanged.

- [ ] **Step 5: Update the actual-state document**

Record that the accepted protocol now includes step 0 and 5k, while clearly distinguishing code capability from server execution status. Do not claim that formal training or new checkpoints already exist.

- [ ] **Step 6: Run documentation contract tests and verify GREEN**

```bash
pytest -q \
  scripts/pi05_libero_bsp_phase1_server_test.py \
  scripts/server_runtime_contract_test.py
```

Expected: all selected contract tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add docs/pi05_libero_bsp_phase1_server.md \
  docs/pi05_libero_bsp_server_state.md \
  scripts/pi05_libero_bsp_phase1_server_test.py
git commit -m "docs: extend phase-one acceptance milestones"
```

### Task 5: Full verification, push, and server handoff

**Files:**
- Verify all modified files from Tasks 1-4
- No new production files

**Interfaces:**
- Produces: a tested commit series on `main`, pushed to `origin/main`
- Produces: a server revision ready for preflight and formal training start

- [ ] **Step 1: Run the complete targeted suite**

```bash
pytest -q \
  src/openpi/training/train_planning_test.py \
  src/openpi/training/data_loader_test.py \
  scripts/train_test.py \
  packages/openpi-client/src/openpi_client/libero_report_test.py \
  scripts/compare_libero_phase1_test.py \
  scripts/pi05_libero_bsp_phase1_server_test.py \
  scripts/server_runtime_contract_test.py
```

Expected: zero failures; third-party warnings may be reported separately but cannot hide a nonzero exit code.

- [ ] **Step 2: Run formatting and repository checks**

```bash
ruff check \
  src/openpi/training/config.py \
  src/openpi/training/train_planning.py \
  src/openpi/training/checkpoints.py \
  scripts/train.py \
  packages/openpi-client/src/openpi_client/libero_report.py \
  scripts/compare_libero_phase1.py
git diff --check
git status --short --branch
```

Expected: no lint errors, no whitespace errors, and only intentional committed history.

- [ ] **Step 3: Audit fixed protocol strings**

```bash
rg -n "exactly six|six fixed|12,000 episodes|12000|只比较 10k、20k、30k|六个 checkpoint" \
  packages/openpi-client/src/openpi_client/libero_report.py \
  scripts/compare_libero_phase1.py \
  docs/pi05_libero_bsp_phase1_server.md \
  docs/pi05_libero_bsp_server_state.md
```

Expected: no stale phase-one protocol matches; unrelated network endpoint wording is outside the searched sections or explicitly reviewed.

- [ ] **Step 4: Push the verified commit series**

```bash
git push origin main
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

Expected: local `HEAD` equals remote `refs/heads/main`.

- [ ] **Step 5: Pull on the server only after both 100-step pilots finish**

In the Web VS Code terminal, confirm no GPU compute process and clean server worktree, then:

```bash
git -C /root/openpi-bsp-work/repo/openpi05-bsp pull --ff-only origin main
git -C /root/openpi-bsp-work/repo/openpi05-bsp rev-parse HEAD
```

Expected: server SHA equals the pushed local/remote SHA. Do not pull while either pilot process is active.

- [ ] **Step 6: Run server lightweight gates**

Use the OpenPI Python 3.11 environment to run the updated planning, report CLI, runtime, and runbook contract tests. Confirm the phase-one configs expose all five permanent steps and that a short isolated test creates `0/params` before any positive-step checkpoint.

- [ ] **Step 7: Start the first formal 30k training process**

After checking GPU idle, dataset/sidecar/norm hashes, new checkpoint/log paths, and at least 80 GiB free, start the selected LoRA Baseline 30k command with seed 42, effective batch 256, micro-batch 64, no EMA, save interval 1k, and offline W&B. Confirm the process is alive, the log identifies the correct config/protocol, and the new `0/params` checkpoint completes before reporting that formal training has started.
