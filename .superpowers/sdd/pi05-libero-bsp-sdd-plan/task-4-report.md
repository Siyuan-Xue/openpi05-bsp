# Task 4 report — BSP serving, deterministic requests, and auditable LIBERO evaluation

## Implementation

- Added `BspLiberoOutputs` and selected it only for `use_bsp` LIBERO data
  configs. OpenPI policy construction already orders model outputs, quantile
  unnormalization, and then data outputs, so the transform receives physical
  BSP parameters. It validates an unbatched 16-row finite result, slices the
  controls-first channels `0:7` plus knot channel `7`, and delegates to the
  author-faithful decoder from Task 1. That decoder preserves equal boundary
  knots, projects only descending knots by `previous + 1e-6`, uses the first
  12 controls, evaluates eight points in `[knots[3], knots[-4]]`, and disables
  extrapolation. Baseline LIBERO output remains the normal horizon-16 7-D
  action sequence.
- Added the dependency-free reserved websocket request key
  `__openpi_inference_seed`. `Policy.infer` copies and pops it before any
  observation transform. A JAX request with the key uses
  `jax.random.key(seed)` without changing the policy's stateful RNG; an absent
  key executes the original `jax.random.split(self._rng)` path exactly.
  PyTorch behavior is unchanged apart from removing the request-envelope key.
- Added dependency-free LIBERO evaluation primitives to `openpi-client`:
  canonical suite resolution, exact episode identities and initial-state
  fingerprints, SHA-256-derived uint32 replan seeds, strict 7-D action
  validation/first-eight selection, typed policy and infrastructure failures,
  infrastructure-only retry, aggregation, manifest validation, incremental
  JSONL and CSV/JSON artifacts, and deterministic first-success/first-failure
  video selection.
- Reworked the LIBERO evaluator while keeping the established
  `task_suite_name` CLI field. It supports only spatial, object, goal, 10, or
  all four (legacy `libero_*` names remain accepted; LIBERO-90 is rejected),
  verifies 10 tasks per suite, uses the official fixed initial-state index and
  a content fingerprint, executes at native 10 Hz, replans every eight steps,
  and supplies the same `(eval seed, suite, task, init state, replan index)`
  flow-noise seed to baseline and BSP.
- Baseline evaluation requires the horizon-16 server output and executes its
  first eight actions. BSP evaluation requires the already-decoded horizon-8
  server output and executes all eight. No alignment, time scaling, async
  planning, or gripper thresholding was added.
- Simulator create/reset/step errors, server-connect/container errors, and
  live websocket/network errors are classified separately at narrow operation
  boundaries. Only those failures are retried, at most twice, after resetting
  the same initial state and seed. Server/model errors, invalid output shape,
  non-finite output, and invalid BSP decoding are counted policy failures and
  are not retried. Exhausted infrastructure attempts are excluded from metric
  denominators and make `acceptance_complete` false; the evaluator writes the
  final summary and exits nonzero.
- Added auditable output files: `manifest.json`, `episodes.jsonl`, `tasks.csv`,
  `suites.csv`, `summary.json`, and deterministic video paths. The manifest
  requires code SHA, dataset revision, nullable baseline/BSP cache hash, norm
  hash, checkpoint identity, container digest, train/eval seeds, selected
  suites/protocol, and every fixed BSP parameter. Episode rows preserve the
  paired key, initial-state fingerprint, status, denominator inclusion,
  attempts/history, steps, replans, and latency samples for later paired
  bootstrap analysis. A partial/non-empty run directory is refused.
- Added an optional websocket connection deadline and public close method for
  evaluator reconnection. The default `connection_timeout=None` retains the
  prior indefinite retry and websocket library handshake timeout behavior.
- Kept all client/evaluator bookkeeping compatible with the existing Python
  3.8 LIBERO container. Shared `openpi-client` modules also parse with Python
  3.7 grammar and do not import NumPy, LIBERO, MuJoCo, JAX, or SciPy.

## Files

- `packages/openpi-client/src/openpi_client/inference.py`
- `packages/openpi-client/src/openpi_client/inference_test.py`
- `packages/openpi-client/src/openpi_client/libero_eval.py`
- `packages/openpi-client/src/openpi_client/libero_eval_test.py`
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- `src/openpi/policies/policy.py`
- `src/openpi/policies/policy_seed_test.py`
- `src/openpi/policies/libero_policy.py`
- `src/openpi/policies/libero_policy_test.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader_test.py`
- `src/openpi/training/bsp_test.py`
- `examples/libero/main.py`
- `.superpowers/sdd/pi05-libero-bsp-sdd-plan/task-4-report.md`

## TDD evidence

The dependency-free request/evaluator tests were added before their production
modules. The genuine RED run failed only because those modules did not exist:

```text
$ PYTHONPATH=packages/openpi-client/src python3 -m unittest \
    openpi_client.inference_test openpi_client.libero_eval_test
EE
...
ImportError: cannot import name 'inference' from 'openpi_client'
...
ImportError: cannot import name 'libero_eval' from 'openpi_client'
...
Ran 2 tests in 0.000s
FAILED (errors=2)
```

The focused local GREEN gate is dependency-free:

```text
$ PYTHONPATH=packages/openpi-client/src python3 -m unittest \
    openpi_client.inference_test openpi_client.libero_eval_test
...............
----------------------------------------------------------------------
Ran 15 tests in 0.002s

OK
```

It covers suite selection, stable request seeds, exact paired episode identity,
same-seed infrastructure retries, exhausted-infrastructure exclusion, policy
failure non-retry, phase classification, baseline/BSP horizon selection and
first-eight execution, malformed/non-finite actions, task/suite/macro
aggregation, withholding the four-suite field for partial suite sets, first
success/failure video naming, complete manifest fields, and emitted
JSONL/CSV/JSON artifacts.

Dependency-heavy tests were also written first for BSP output decoding/config
selection and JAX stateless/stateful RNG behavior. The local host intentionally
has no project pytest/NumPy/SciPy/JAX runtime, and the required attempt stopped
before collection:

```text
$ PYTHONPATH=src:packages/openpi-client/src python3 -m pytest \
    src/openpi/policies/libero_policy_test.py \
    src/openpi/policies/policy_seed_test.py \
    src/openpi/training/bsp_test.py \
    src/openpi/training/data_loader_test.py \
    packages/openpi-client/src/openpi_client/inference_test.py \
    packages/openpi-client/src/openpi_client/libero_eval_test.py
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

Required server GREEN gate:

```text
uv run pytest \
  packages/openpi-client/src/openpi_client/inference_test.py \
  packages/openpi-client/src/openpi_client/libero_eval_test.py \
  src/openpi/policies/libero_policy_test.py \
  src/openpi/policies/policy_seed_test.py \
  src/openpi/training/bsp_test.py \
  src/openpi/training/data_loader_test.py

uv run ruff check \
  packages/openpi-client/src/openpi_client/inference.py \
  packages/openpi-client/src/openpi_client/inference_test.py \
  packages/openpi-client/src/openpi_client/libero_eval.py \
  packages/openpi-client/src/openpi_client/libero_eval_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy.py \
  src/openpi/policies/policy.py src/openpi/policies/policy_seed_test.py \
  src/openpi/policies/libero_policy.py src/openpi/policies/libero_policy_test.py \
  src/openpi/training/config.py src/openpi/training/data_loader_test.py \
  src/openpi/training/bsp_test.py examples/libero/main.py
```

The evaluator must additionally pass the planned container EGL smoke and
official-checkpoint diagnostic before any acceptance run. No simulator result
is claimed locally.

## Static verification

```text
$ python3 -m py_compile <all Task 4 Python files>
exit 0; no output

$ ast.parse(..., feature_version=(3, 8))  # evaluator/client files
python3.8 syntax OK ...

$ ast.parse(..., feature_version=(3, 7))  # shared openpi-client files
python3.7 syntax OK ...

$ git diff --check
exit 0; no output
```

No dependency installation/sync, data or checkpoint download, server launch,
simulation, training, inference, or heavyweight test was performed.

## Self-review

- Confirmed BSP decoding is selected by training config semantics, not by an
  evaluator-side spline implementation, and remains after unnormalization.
- Confirmed explicit JAX seeds leave `_rng` byte-for-byte unchanged while the
  absent-seed branch still returns the same two `jax.random.split` outputs.
- Confirmed the reserved key is copied/popped before all observation transforms
  and never reaches `Observation.from_dict`.
- Confirmed every retry resets/reseeds the same official initial state and
  restarts replan index zero, so all explicit model seeds repeat exactly.
- Confirmed broad exception handling is limited to one labeled external phase
  (connect, infer, simulator create/reset/step) or best-effort resource cleanup;
  no catch surrounds the whole rollout, aggregation, or artifact write.
- Confirmed policy and timeout failures enter success-rate denominators;
  exhausted infrastructure does not. Transient infrastructure history is
  retained even when a later attempt completes.
- Confirmed a suite macro is always labeled as such, while
  `four_suite_macro_success_rate` remains null unless all four suites have a
  valid suite denominator.
- Confirmed only the first counted success and first policy/timeout failure per
  task can reserve a video, and paths include suite, task, outcome, initial
  state, and episode identity.
- Confirmed episode records release replay frames after optional video writing,
  avoiding retention of every rollout image in the 12,000-episode run.

## Concerns / server follow-up

- SciPy/NumPy/JAX policy output and key behavior require the server gate above;
  Task 5 owns the direct SciPy lock and environment creation.
- The locked `websockets` version must exercise the new finite connection
  deadline and `recv(timeout=...)` path in the container. Default clients do
  not take that path and retain their existing behavior.
- Manifest checkpoint, norm, cache, and container identities are deliberately
  explicit CLI inputs. Task 5's runbook must compute/pass their real hashes and
  digest; placeholders are rejected.
- The previously reported Task 1/2 episode-local knot subtraction issue is
  intentionally outside Task 4. The decoder is shift-invariant and consumes
  the unnormalized `[16,8]` parameters it receives without changing cache
  semantics.

## Independent-review fix round 1

The independent Task 4 review found two Important and three Minor issues. This
round addresses all five without changing Task 1/2 data semantics or entering
Task 5 scope.

### Fixes

- Added `inference_timeout` to `WebsocketClientPolicy`, independently from the
  connection/metadata deadline. Its default remains `None`, which preserves
  the existing unbounded `recv()` behavior for every existing client. The
  evaluator passes a finite default of 120 seconds and the client calls
  `recv(timeout=...)`; a resulting `TimeoutError` follows the already tested
  network-infrastructure classification, connection invalidation, identical
  seed, and at-most-two-retry path. Both deadlines and retry count are now in
  the manifest.
- Added a strict policy-protocol resolver and explicit
  `expected_action_horizon` evaluator option. Phase-one baseline defaults to
  `baseline_h16`, the official `pi05_libero` calibration checkpoint can select
  `baseline_h10_calibration` with horizon 10, and BSP is fixed to
  `bsp_decoded_h8`. Other baseline horizons and any non-8 BSP decoded horizon
  are rejected. The protocol name, expected server horizon, and fixed executed
  horizon 8 are recorded and cross-validated in the manifest.
- Added fake-based central wiring contracts for the real evaluator path: the
  exact initial-state object is reused, the reserved derived seed reaches the
  websocket request, a timeout invalidates the connection, retry resets the
  same state and reuses the same seed, and `_ClientHolder` passes a finite
  inference deadline. Added a `Policy.infer` integration contract proving the
  reserved field is absent before the first observation transform while the
  caller's request remains unchanged.
- Reordered episode/video publication. The completed episode JSONL row is now
  appended before video selection or ffmpeg. Video failures are caught only
  around encoding, appended to independent `artifact_errors.jsonl`, counted in
  `summary.json`, and make `acceptance_complete=false`; the rollout result is
  retained and policy metrics remain unchanged. A focused call-order test
  requires `episode -> video -> artifact_error`.
- Tightened `BspLiberoOutputs` to the fixed model contract `(16,32)` before it
  extracts the `[16,8]` controls/knot payload. Tests now reject `(16,9)` and
  `(16,31)` in addition to wrong rows, too-few channels, and non-finite data.

### Review-fix TDD evidence

The dependency-free contracts for policy protocol and artifact auditing were
added before their implementations. The focused RED run failed at the missing
production symbols:

```text
$ PYTHONPATH=packages/openpi-client/src python3 -m unittest \
    openpi_client.inference_test openpi_client.libero_eval_test
....E........E...
...
AttributeError: module 'openpi_client.libero_eval' has no attribute 'ArtifactError'
...
AttributeError: module 'openpi_client.libero_eval' has no attribute 'resolve_policy_protocol'
...
Ran 17 tests in 0.003s
FAILED (errors=2)
```

After implementation, the dependency-free GREEN gate is:

```text
$ PYTHONPATH=packages/openpi-client/src python3 -m unittest \
    openpi_client.inference_test openpi_client.libero_eval_test
.................
----------------------------------------------------------------------
Ran 17 tests in 0.005s

OK
```

The fake websocket/evaluator, NumPy/SciPy BSP, and JAX policy integration tests
are written but remain server-only because the local coding host has no pytest
or project dependencies:

```text
$ PYTHONPATH=src:packages/openpi-client/src python3 -m pytest \
    packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
    scripts/libero_eval_test.py \
    src/openpi/policies/libero_policy_test.py \
    src/openpi/policies/policy_seed_test.py
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

Required focused server gate for this review round:

```text
uv run pytest \
  packages/openpi-client/src/openpi_client/inference_test.py \
  packages/openpi-client/src/openpi_client/libero_eval_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
  scripts/libero_eval_test.py \
  src/openpi/policies/libero_policy_test.py \
  src/openpi/policies/policy_seed_test.py
```

Static `py_compile`, Python 3.7 client grammar, Python 3.8 evaluator grammar,
and `git diff --check` remain green. No dependency install/sync, network
download, policy server, simulator, training, or inference was run locally.

Remaining server-only gates are the real locked-websockets timeout behavior,
the JAX transform/RNG integration, SciPy BSP decoding, an official horizon-10
calibration smoke, a horizon-16 baseline smoke, and a decoded-horizon-8 BSP
smoke in LIBERO/EGL.
