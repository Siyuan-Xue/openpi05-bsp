# Task 3 report — JAX gradient accumulation and exact checkpoints

## Implementation

- Added `TrainConfig.micro_batch_size` as an optional **global** JAX
  micro-batch size. `batch_size` remains the effective global batch. The JAX
  loader receives the global micro-batch and retains its existing
  `batch / jax.process_count()` local partitioning; PyTorch loading continues
  to use `batch_size`.
- Added dependency-free planning helpers that validate positive sizes,
  effective-batch/micro-batch divisibility, global device divisibility,
  process divisibility, and device/process topology. `None` and an explicit
  micro-batch equal to the effective batch both plan one accumulation.
- Split the JAX update into a jitted micro-gradient computation, a donated
  jitted tree accumulator, and one jitted optimizer application. Consecutive
  loader micro-batches are processed without stacking a global batch in device
  memory. Losses and gradient leaves are averaged before the single
  `tx.update`, so the optimizer chain's global-norm clip sees the averaged
  gradient. `state.step` and EMA advance once in that optimizer application.
- Folded the training RNG by the current optimizer step and, when accumulating,
  by the accumulation index. The one-batch path retains the exact legacy
  step-folded key for backward-compatible no-accumulation behavior.
- Changed the JAX loop to enumerate **updated** optimizer-step numbers. A
  checkpoint predicate selects completed multiples of `save_interval` plus the
  final step. Immediately before saving, the realized `train_state.step` is
  synchronized and checked against the directory label, then that realized
  value is passed to Orbax. No accumulator exists in `TrainState`, and saves
  occur only after `apply_optimizer_step` returns.
- Completed both H16 LIBERO recipes with pi05, action dimension 32, horizon 16,
  continuous state input, seed 42, effective batch 256, global micro-batch 1,
  the official `pi05_base` JAX loader, the specified cosine schedule,
  AdamW/global-norm clip 1.0, EMA 0.999, 30,000 optimizer steps, save interval
  1,000, and keep period 10,000. Their Task 2 dataset, asset, norm, and BSP
  distinctions are unchanged. The stale PyTorch placeholder path was removed
  only from these two explicitly JAX recipes.
- Updated `debug_restore` from checkpoint `9` to checkpoint `10`; this is the
  one existing config reference directly coupled to the corrected JAX
  post-update checkpoint labels. Default values and unrelated experiment
  recipes were otherwise left unchanged.

With `max_to_keep=1`, the unchanged checkpoint manager keeps the current latest
save while `keep_period=10_000` preserves the exact H16 milestone directories
`10000`, `20000`, and `30000`.

## Files

- `scripts/train.py`
- `scripts/train_test.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`
- `src/openpi/training/data_loader_test.py`
- `src/openpi/training/train_planning.py`
- `src/openpi/training/train_planning_test.py`
- `.superpowers/sdd/pi05-libero-bsp-sdd-plan/task-3-report.md`

## TDD evidence

The dependency-free planning test was added before its module. The genuine RED
run failed for the intended missing production module:

```text
$ PYTHONPATH=src python3 -m unittest openpi.training.train_planning_test
E
======================================================================
ERROR: train_planning_test (unittest.loader._FailedTest.train_planning_test)
----------------------------------------------------------------------
ImportError: Failed to import test module: train_planning_test
...
ModuleNotFoundError: No module named 'openpi.training.train_planning'

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (errors=1)
```

The focused GREEN run after the minimal helper implementation is:

```text
$ PYTHONPATH=src python3 -m unittest openpi.training.train_planning_test
..........
----------------------------------------------------------------------
Ran 10 tests in 0.007s

OK
```

Those tests cover default/equal no-accumulation planning, the single-H20
256-to-1 plan, global-to-local multi-process geometry, invalid sizes and
topologies, leafwise gradient sum/average, nonpositive average counts, many
micro-batches producing one completed optimizer step, exact interval/final
checkpoint labels, the 10k/20k/30k milestones, and resume starting at 10001
after a completed 10000 checkpoint.

Before the trainer/config/data-loader implementation, dependency-heavy
contracts were also added to:

- make the debug trainer accumulate two micro-batches per optimizer step;
- require final checkpoint directories `2` and `4`, never off-by-one `1` or
  `3`, across a resume;
- require the JAX loader to emit the configured global micro-batch; and
- bind both H16 recipes to the shared full-finetuning/checkpoint recipe while
  retaining their distinct data semantics.

During self-review, an additional server-only equivalence contract was added to
require `micro_batch_size=None` and `micro_batch_size=batch_size` training to
produce exactly equal parameter leaves. It covers the explicit legacy-path
compatibility check; unlike the production-driving contracts above, it was
added as follow-up coverage and is not presented as test-first evidence.

The local host intentionally has no pytest/JAX/NumPy project runtime, and
installation or sync was prohibited. The attempted focused command therefore
stopped before collection as expected:

```text
$ PYTHONPATH=src python3 -m pytest scripts/train_test.py src/openpi/training/data_loader_test.py
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

Required server GREEN/equivalence gate:

```text
uv run pytest \
  src/openpi/training/train_planning_test.py \
  scripts/train_test.py \
  src/openpi/training/data_loader_test.py
```

No local JAX, optimizer, Orbax, or training result is claimed.

## Static verification

```text
$ python3 -m py_compile \
  scripts/train.py scripts/train_test.py \
  src/openpi/training/config.py \
  src/openpi/training/data_loader.py \
  src/openpi/training/data_loader_test.py \
  src/openpi/training/train_planning.py \
  src/openpi/training/train_planning_test.py
exit 0; no output

$ python3 -m compileall -q \
  scripts/train.py scripts/train_test.py \
  src/openpi/training/config.py \
  src/openpi/training/data_loader.py \
  src/openpi/training/data_loader_test.py \
  src/openpi/training/train_planning.py \
  src/openpi/training/train_planning_test.py
exit 0; no output

$ git diff --check
exit 0; no output
```

No dependency install, repository sync, data download, simulator, training run,
or heavy test was performed.

## Self-review

- Confirmed the loader consumes one micro-batch at a time; it does not stack
  256 samples before the jitted loss, so activation-memory savings are real.
- Confirmed every micro-gradient observes the same pre-update parameters and
  optimizer step, while accumulation indices are consecutive and RNG-distinct.
- Confirmed summed gradients are divided before the only `state.tx.update`
  call; clipping remains inside the existing optimizer chain and therefore
  happens after averaging.
- Confirmed `state.step`, optimizer state, parameters, and EMA change only in
  `apply_optimizer_step`, once per effective batch.
- Confirmed metrics report the averaged loss and averaged-gradient norm.
- Confirmed saves are below the optimizer application in the host loop, use the
  updated state value as their label, and cannot capture a partial accumulator.
- Confirmed resume derives its next step from restored `train_state.step`, not
  a partial micro-batch counter or a zero-based loop index.
- Confirmed the default `micro_batch_size=None`, checkpoint manager defaults,
  PyTorch loader behavior, and unrelated experiment recipes remain intact.

## Concerns / server follow-up

The server must exercise Flax NNX gradient-tree sharding/filter structure,
donated accumulator buffers, Optax state advancement, EMA equivalence, and
Orbax async save/restore with real project dependencies. In particular, the
debug integration/equivalence tests are the required proof that the separated
micro-gradient and optimizer jits agree with the legacy single-batch path and
that checkpoint directories round-trip at exact updated steps.

For the real H20 job, keep the committed global micro-batch at 1 for the safe
first run. Probe 1/2/4/8 in separate processes and override **both** H16 configs
to the same largest stable value only after memory evidence; effective batch
256 and optimizer-step/checkpoint semantics remain unchanged.

## Review fix — real accumulated-update equivalence coverage

Review identified that the original runtime equivalence test compared
`micro_batch_size=None` with `micro_batch_size=batch_size`. Both plan one
micro-batch, so that test covered legacy-path equivalence but never exercised
gradient accumulation. The planning test likewise proved only indices and step
labels, not optimizer behavior.

Added
`test_two_micro_batches_match_one_direct_batch_and_advance_state_once` in
`scripts/train_test.py`. It uses a two-parameter NNX linear model whose loss
deliberately ignores RNG, a four-example global batch, and a real production
plan with global `micro_batch_size=2 < batch_size=4`. The test:

- computes the direct large-batch gradient through
  `train.compute_microbatch_grad`;
- computes two consecutive production micro-gradients against the same
  pre-update state and combines them through `train.add_microbatch_results`;
- compares the averaged accumulated gradient tree to the direct gradient tree;
- calls `train.apply_optimizer_step` once for each path with a real Optax Adam
  state;
- compares updated parameter, optimizer-state, EMA, loss, and gradient-norm
  trees between the two paths; and
- independently asserts a nonzero gradient/parameter change, optimizer count
  `0 -> 1`, train step `7 -> 8`, and exactly one EMA interpolation from the old
  EMA tree to the new parameters.

This is not a source/shape assertion: it invokes the three production
accumulation/application boundaries directly. Removing gradient averaging,
updating per micro-batch, advancing EMA or optimizer state more than once, or
incrementing `TrainState.step` per micro-batch makes the contract fail.

The review fix is test-only because the production accumulation path already
implements the required behavior. The new contract was written before any
production edit; no production weakening or accommodation was made. Runtime
RED/GREEN execution is explicitly deferred because the local coding host still
lacks pytest/JAX/NumPy/Flax/Optax and dependency installation is prohibited:

```text
$ PYTHONPATH=src python3 -m pytest scripts/train_test.py \
    -k two_micro_batches_match_one_direct_batch_and_advance_state_once
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

Required focused server gate:

```text
uv run pytest scripts/train_test.py \
  -k two_micro_batches_match_one_direct_batch_and_advance_state_once
```

Local static parsing remains available and is included in the final fix
verification. The server test is deterministic: RNG keys differ between the
direct and accumulated paths by design, while the tiny loss ignores RNG so the
mathematical gradient comparison is exact up to float32 tolerance.
