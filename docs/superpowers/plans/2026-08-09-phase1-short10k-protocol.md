# π0.5 + LIBERO BSP Phase-One Short 10k Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the phase-one 30k protocol with an auditable 10k protocol whose fixed milestones are 0/1k/2k/5k/10k, then restart the server baseline LoRA run under a new experiment identity.

**Architecture:** Keep checkpoint creation and report generation mechanisms unchanged; replace the single phase-one protocol definition consistently at the configuration, report, CLI, contract-test, and runbook layers. Preserve the abandoned 30k server run as immutable evidence, and launch the new run from `pi05_base` in a collision-free directory.

**Tech Stack:** Python 3.11, JAX, Orbax, unittest/pytest, Ruff, Git worktrees, Bash, Chrome-controlled Alibaba DSW terminal.

## Global Constraints

- Phase-one training ends at exactly 10,000 optimizer steps.
- Fixed milestones are exactly `(0, 1_000, 2_000, 5_000, 10_000)` for all four phase-one configs.
- `save_interval=1_000`, `keep_period=10_000`, seed 42, effective batch 256, and the existing optimizer/loss remain unchanged.
- The deployed H20 route is LoRA, micro-batch 64, EMA disabled, and W&B offline.
- Baseline and BSP each evaluate five checkpoints; the report still contains 10 runs and 20,000 episodes.
- Preserve the abandoned `phase1-seed42-baseline` run and all its logs/checkpoints; never overwrite or resume it.
- New experiment names are `phase1-short10k-seed42-baseline` and `phase1-short10k-seed42-bsp`.
- Orbax writes only to `/root`; persistent archives may be written only below `/mnt/data/siyuanxue`.
- Do not run two GPU jobs concurrently and do not use `SIGKILL` if graceful termination fails.

---

### Task 1: Gracefully stop and audit the superseded 30k server run

**Files:**
- Create on server: `/root/openpi-bsp-work/experiments/logs/protocol-transition-30k-to-10k-<timestamp>.txt`
- Preserve: `/root/openpi-bsp-work/experiments/checkpoints/pi05_libero_baseline_lora_h16/phase1-seed42-baseline/`
- Preserve: `/root/openpi-bsp-work/experiments/logs/train-phase1-seed42-baseline.log`

**Interfaces:**
- Consumes: PID from `train-phase1-seed42-baseline.pid` and the approved short10k design.
- Produces: an exited old process, a free GPU, and an immutable audit record containing PID, command, last step, checkpoint list, error count, code SHA, and termination reason.

- [ ] **Step 1: Verify the process identity without mutation**

```bash
WORK=/root/openpi-bsp-work
PID="$(cat "$WORK/experiments/logs/train-phase1-seed42-baseline.pid")"
ps -p "$PID" -o pid=,etime=,args=
ps -p "$PID" -o args= | grep -F 'scripts/train.py pi05_libero_baseline_lora_h16'
ps -p "$PID" -o args= | grep -F -- '--exp-name phase1-seed42-baseline'
```

Expected: both identity checks return zero and show the old 30k run.

- [ ] **Step 2: Capture pre-termination evidence**

```bash
LOG="$WORK/experiments/logs/train-phase1-seed42-baseline.log"
RUN="$WORK/experiments/checkpoints/pi05_libero_baseline_lora_h16/phase1-seed42-baseline"
grep -E 'Progress on:|Step [0-9]+:' "$LOG" | tail -n 20
find "$RUN" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n
grep -Eic 'Traceback|out of memory|RESOURCE_EXHAUSTED|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG"
```

Expected: finite training evidence and an abnormal-error count of zero.

- [ ] **Step 3: Send SIGTERM and wait for graceful exit**

```bash
kill -TERM "$PID"
for attempt in $(seq 1 60); do
  ps -p "$PID" >/dev/null 2>&1 || break
  sleep 1
done
ps -p "$PID" >/dev/null 2>&1 && echo 'STOP: graceful termination failed' >&2 && exit 2
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

Expected: process exits within 60 seconds and no compute PID remains. Do not escalate to `SIGKILL`.

- [ ] **Step 4: Atomically write the audit record**

Use a unique UTC timestamp, write to a `.partial` file below the logs directory, include the exact reason
`approved protocol change from 30k to 10k`, then rename it to `.txt`. Verify with `test -s` and `sha256sum`.

### Task 2: Establish the isolated implementation worktree

**Files:**
- Create: `.worktrees/phase1-short10k/`
- Branch: `feat/phase1-short10k`

**Interfaces:**
- Consumes: clean local `main` containing design and plan commits.
- Produces: an isolated named branch based on `main`.

- [ ] **Step 1: Confirm local repository state**

```bash
git status --short --branch
git rev-parse HEAD
git check-ignore -q .worktrees
```

Expected: local `main` is clean and `.worktrees` is ignored.

- [ ] **Step 2: Create the worktree**

```bash
git worktree add .worktrees/phase1-short10k -b feat/phase1-short10k main
git -C .worktrees/phase1-short10k status --short --branch
```

Expected: a clean named branch at the plan commit.

### Task 3: RED — define the configuration and retention contract

**Files:**
- Modify: `src/openpi/training/data_loader_test.py`
- Modify: `src/openpi/training/train_planning_test.py`

**Interfaces:**
- Consumes: `TrainConfig.num_train_steps`, `TrainConfig.permanent_checkpoint_steps`, and `checkpoint_should_keep`.
- Produces: failing tests that require the exact short10k protocol.

- [ ] **Step 1: Change the phase-one config expectation**

```python
expected_milestones = (0, 1_000, 2_000, 5_000, 10_000)
assert config.num_train_steps == 10_000
assert config.permanent_checkpoint_steps == expected_milestones
```

- [ ] **Step 2: Change the retention test**

Test candidates `(0, 1_000, 2_000, 3_000, 5_000, 9_000, 10_000)` and require the permanent subset
`[0, 1_000, 2_000, 5_000, 10_000]`.

- [ ] **Step 3: Run RED tests**

```bash
python -m pytest -q src/openpi/training/data_loader_test.py src/openpi/training/train_planning_test.py
```

Expected: failures show production still contains 30k and the old milestones.

### Task 4: GREEN — update all four phase-one configs

**Files:**
- Modify: `src/openpi/training/config.py:810-925`
- Test: `src/openpi/training/data_loader_test.py`
- Test: `src/openpi/training/train_planning_test.py`

**Interfaces:**
- Consumes: the RED expectations from Task 3.
- Produces: four configs with `num_train_steps=10_000` and `permanent_checkpoint_steps=(0, 1_000, 2_000, 5_000, 10_000)`.

- [ ] **Step 1: Apply the minimal config change**

Change only the four LIBERO phase-one Baseline/BSP full/LoRA configs. Do not change the `TrainConfig`
default or unrelated ALOHA/DROID/debug configs.

- [ ] **Step 2: Run GREEN tests**

```bash
python -m pytest -q src/openpi/training/data_loader_test.py src/openpi/training/train_planning_test.py
```

Expected: PASS.

- [ ] **Step 3: Commit the config contract**

```bash
git add src/openpi/training/config.py src/openpi/training/data_loader_test.py src/openpi/training/train_planning_test.py
git commit -m 'feat: shorten phase-one training protocol to 10k'
```

### Task 5: RED/GREEN — migrate the comparison report

**Files:**
- Modify: `packages/openpi-client/src/openpi_client/libero_report_test.py`
- Modify: `scripts/compare_libero_phase1_test.py`
- Modify: `packages/openpi-client/src/openpi_client/libero_report.py`
- Modify: `scripts/compare_libero_phase1.py`

**Interfaces:**
- Consumes: manifests whose `checkpoint_step` is one of five short10k milestones.
- Produces: `libero_report.MILESTONES == (0, 1000, 2000, 5000, 10000)` and a ten-run/20,000-episode report.

- [ ] **Step 1: Write failing report tests**

Set `_STEPS = (0, 1000, 2000, 5000, 10000)`. Add or retain an explicit rejection test for step 20,000.
Update CLI help and success-message expectations to `0k, 1k, 2k, 5k, and 10k`.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q packages/openpi-client/src/openpi_client/libero_report_test.py scripts/compare_libero_phase1_test.py
```

Expected: failures reference old milestones and CLI text.

- [ ] **Step 3: Apply minimal report implementation**

```python
MILESTONES = (0, 1000, 2000, 5000, 10000)
```

Keep `TOTAL_EPISODES = EPISODES_PER_RUN * 2 * len(MILESTONES)` and the six artifact files unchanged.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest -q packages/openpi-client/src/openpi_client/libero_report_test.py scripts/compare_libero_phase1_test.py
git add packages/openpi-client/src/openpi_client/libero_report.py \
  packages/openpi-client/src/openpi_client/libero_report_test.py \
  scripts/compare_libero_phase1.py scripts/compare_libero_phase1_test.py
git commit -m 'feat: report short10k phase-one milestones'
```

### Task 6: RED/GREEN — synchronize server contracts and runbooks

**Files:**
- Modify: `scripts/pi05_libero_bsp_phase1_server_test.py`
- Modify: `scripts/server_runtime_contract_test.py`
- Modify: `docs/pi05_libero_bsp_phase1_server.md`
- Modify: `docs/pi05_libero_bsp_server_state.md`

**Interfaces:**
- Consumes: short10k config/report constants.
- Produces: commands and tests using 0/1k/2k/5k/10k and the new experiment names.

- [ ] **Step 1: Change contract expectations first**

Require the new steps, `--num-train-steps 10000`, and A/B evaluation calls at 0, 1000, 2000, 5000,
and 10000. Reject live first-phase commands that still use 20000 or 30000.

- [ ] **Step 2: Run RED contract tests**

```bash
python -m pytest -q scripts/pi05_libero_bsp_phase1_server_test.py scripts/server_runtime_contract_test.py
```

Expected: failures identify old runbook values.

- [ ] **Step 3: Update the runbooks**

Use these exact values in live commands:

```bash
export BASELINE_EXP=phase1-short10k-seed42-baseline
export BSP_EXP=phase1-short10k-seed42-bsp
--num-train-steps 10000
```

Preserve historical statements that explicitly describe superseded runs and keep 20,000 total episodes.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest -q scripts/pi05_libero_bsp_phase1_server_test.py scripts/server_runtime_contract_test.py
git add scripts/pi05_libero_bsp_phase1_server_test.py scripts/server_runtime_contract_test.py \
  docs/pi05_libero_bsp_phase1_server.md docs/pi05_libero_bsp_server_state.md
git commit -m 'docs: migrate phase-one runbook to short10k'
```

### Task 7: Verify, integrate, and publish

**Files:**
- Verify all files modified in Tasks 3-6.

**Interfaces:**
- Consumes: completed feature branch.
- Produces: green local `main` and matching `origin/main`.

- [ ] **Step 1: Run targeted verification**

```bash
python -m pytest -q \
  src/openpi/training/data_loader_test.py src/openpi/training/train_planning_test.py \
  packages/openpi-client/src/openpi_client/libero_report_test.py scripts/compare_libero_phase1_test.py \
  scripts/pi05_libero_bsp_phase1_server_test.py scripts/server_runtime_contract_test.py
ruff check src/openpi/training/config.py src/openpi/training/data_loader_test.py \
  src/openpi/training/train_planning_test.py packages/openpi-client/src/openpi_client/libero_report.py \
  packages/openpi-client/src/openpi_client/libero_report_test.py scripts/compare_libero_phase1.py \
  scripts/compare_libero_phase1_test.py scripts/pi05_libero_bsp_phase1_server_test.py \
  scripts/server_runtime_contract_test.py
git diff --check
```

Expected: all commands return zero.

- [ ] **Step 2: Merge and verify main**

Fast-forward local `main` to `feat/phase1-short10k`, rerun the targeted suite on `main`, and confirm clean status.

- [ ] **Step 3: Push main**

```bash
git push origin main
git ls-remote origin HEAD
```

Expected: local `main`, `origin/main`, and remote HEAD are identical.

### Task 8: Pull, verify, and launch the new server run

**Files:**
- Create: `/root/openpi-bsp-work/experiments/logs/train-phase1-short10k-seed42-baseline.log`
- Create: `/root/openpi-bsp-work/experiments/logs/train-phase1-short10k-seed42-baseline.pid`
- Create: `/root/openpi-bsp-work/experiments/checkpoints/pi05_libero_baseline_lora_h16/phase1-short10k-seed42-baseline/`

**Interfaces:**
- Consumes: published `origin/main`, norm assets, LIBERO v2.0, cached `pi05_base`, and successful LoRA pilot parameters.
- Produces: a running short10k baseline with a complete step 0 checkpoint.

- [ ] **Step 1: Pull exact server commit**

```bash
git -C /root/openpi-bsp-work/repo/openpi05-bsp -c http.version=HTTP/1.1 pull --ff-only origin main
```

Retry the same HTTP/1.1 command only for a transient transport failure; never reset the server worktree.

- [ ] **Step 2: Run lightweight gates**

Verify SHA, clean status, config tuple/end step, report milestones, norm/data inputs, GPU idle, `/root >= 80 GiB`,
and collision-free paths. Run the six targeted test files from Task 7 with server OpenPI Python.

- [ ] **Step 3: Start the new baseline once**

Run `scripts/train.py pi05_libero_baseline_lora_h16` with exp name `phase1-short10k-seed42-baseline`, seed 42,
batch 256, micro-batch 64, `--num-train-steps 10000`, `--save-interval 1000`, `--ema-decay None`, W&B
offline, and the existing dataset/assets/checkpoint roots.

- [ ] **Step 4: Verify start and monitoring**

Confirm process alive, GPU allocated, complete `0/params`, `0/train_state`, and `0/assets`, no Orbax temp
directory, no abnormal log matches, and safe `/root` capacity. Replace the hourly monitor with the new 10k paths.
