# Refactor Async Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Tests are authored before production changes but are not executed on this macOS checkout; the user requires all execution to happen later on the server, after this branch is pushed.

**Goal:** Semantically port the validated video-timing feature and resumable-loader fix into the slim refactor branch, then add BSP and baseline asynchronous inference without restoring code removed by slimming.

**Architecture:** Keep the refactor branch as the only implementation line. A single-owner WebSocket worker handles at most one in-flight request, while pure schedulers own action-plan timing. BSP uses continuous raw spline parameters without segment alignment; baseline async uses RTC guidance inside the OpenPI flow sampler.

**Tech Stack:** Python 3.8 client, Python 3.11 server/model, JAX, NumPy, synchronous websockets, pytest contracts, msgpack.

## Global Constraints

- Work only in the isolated `codex/refactor-async-integration` worktree based on `8420b70`.
- Do not merge `main`; do not restore Docker, Compose, deleted runtime tests, or unrelated dependencies removed by refactor.
- Port feat intent from `7e1ed7a..b12650c`; record every source commit as implemented, superseded, or intentionally omitted.
- Cherry-pick Fix commits `6e650da`, `9bdc18e`, `a40c1a8`, `c4d90be` in order; adapt tests to pytest without importing main-only structure.
- The client remains Python 3.8 compatible and supports websockets 13.1 and 15.x.
- One socket owner, one outstanding request, no concurrent `recv`, no policy inference concurrency, and no daemon worker.
- LIBERO control is wall-clock 20 Hz; video is 40 fps; no catch-up, zero action, repeated delta action, or curve extrapolation.
- BSP uses speedup 1 and no segment alignment for delta-EFF commands.
- RTC uses H=16, s_min=8, n=5, beta=5, delay history size 10, and constrains only the first seven of 32 model action dimensions.
- Latency calibration is 5 warmups plus 20 measurements; nearest-rank p95, not a fixed 60 ms budget.
- Preserve schema-v2/v3 compatibility; all four new modes use schema-v4.
- Do not install environments or run tests on this Mac. Write tests first, record them as unexecuted, commit, push only the refactor branch, then stop. Do not operate the server or update main.

---

### Task 1: Semantic branch integration

**Files:** feat-touched LIBERO evaluator/report/timing files, slim host contracts, training loader-resume files, and a new port manifest under `docs/superpowers/specs/`.

**Interfaces:** Produces the validated schema-v3/video-timing foundation and loader-resume API used by later tasks.

- [x] Compare each feat commit with the slim equivalent and write a source-to-port manifest before changing production code.
- [x] Add the standalone timing contracts and their existing feature tests.
- [x] Integrate schema-v3, video audit, runtime identity, failure-video, and Git identity behavior into refactor's `libero_artifacts.py` architecture.
- [x] Keep slim host-only files deleted and migrate any surviving contract assertions into retained tests.
- [x] Cherry-pick the four Fix commits in order; resolve `train_planning_test.py` in pytest style.
- [x] Review the resulting diff for main-only files or dependencies and commit the semantic port.

### Task 2: Single-owner asynchronous inference worker

**Files:** create `packages/openpi-client/src/openpi_client/async_inference.py` and its test; modify the WebSocket policy only for explicit lifecycle hooks needed by the worker.

**Interfaces:** Produce immutable `InferenceJob`, `InferenceOutcome`, and `AsyncInferenceWorker.submit/poll/wait/reset_generation/close` APIs.

- [x] Write deterministic gate-driven tests for initial blocking, one in-flight request, immutable observations, stale generations, error propagation, reset races, and bounded idempotent shutdown.
- [x] Implement a non-daemon worker with a one-slot queue and exclusive client ownership.
- [x] Make reconnect waits interruptible and treat close-induced connection errors as cancellation only during shutdown or stale generations.
- [x] Add localhost loopback tests for the shared websockets 13.1/15.x API surface without external network access.
- [x] Commit the worker separately.

### Task 3: BSP continuous action plans

**Files:** BSP LIBERO output transform, a client-side spline module and tests, and optional response validation helpers.

**Interfaces:** Preserve `actions:(8,7)` and add `bsp={schema_version, parameters:(16,8), origin_hz:10, degree:3, speedup:1, alignment:"disabled_delta_eff"}`.

- [x] Write tests proving legacy actions remain unchanged and raw parameters are captured after unnormalization but before server decode.
- [x] Write hand-derived spline tests for knot repair, malformed/nonfinite parameters, boundary sampling, and agreement with the existing eight-point decoder.
- [x] Implement pure NumPy continuous spline evaluation suitable for Python 3.8 clients.
- [x] Implement synchronous and calibrated-prefetch BSP schedulers with immediate swap and no segment alignment.
- [x] Commit BSP protocol and scheduling separately.

### Task 4: Baseline RTC in OpenPI flow space

**Files:** Pi0 sampler/model tests, policy/server request-context plumbing, and client RTC plan state.

**Interfaces:** Optional request context carries normalized previous `(16,32)` actions, `s`, and `d`; response preserves `actions:(16,7)` and adds `rtc.model_actions:(16,32)`.

- [x] Write numerical tests mapping the paper's tau=0-to-1 equations onto OpenPI's t=1-to-0 sampler.
- [x] Write tests for H/s/d constraints, the soft mask, first-seven-dimension guidance, zero padding mask, delay history, and deterministic seeds.
- [x] Add an optional RTC context to inference without changing legacy sampler output when absent.
- [x] Implement guided vector-Jacobian-product correction at every denoising step with n=5 and beta=5.
- [x] Return opaque normalized model actions to the client and immediately swap chunks per Algorithm 1.
- [x] Commit RTC separately.

### Task 5: 20 Hz control, calibration, and schema-v4

**Files:** `examples/libero/main_v4.py`, client v4 evaluator/report/timing modules, and their tests.

**Interfaces:** Add modes `baseline_sync_n5`, `baseline_rtc`, `bsp_spline_sync`, `bsp_spline_async`; schema-v4 separates request, activation, latency, stall, and underflow events.

- [x] Write manual-clock tests for 50 ms deadlines, no catch-up, initial synchronous plan, overlap without stalls, and underflow pause/resume.
- [x] Add 5+20 latency calibration, nearest-rank p95, 50 ms BSP budget rounding, RTC d_init, and immutable calibration fingerprinting.
- [x] Integrate the worker and family-specific schedulers into the LIBERO loop while keeping v2/v3 readers unchanged.
- [x] Add strict v4 serialization, validation, report comparison, 40 fps stall frames, and fail-closed malformed-event handling.
- [x] Commit control integration and artifact schema separately.

### Task 6: Documentation, static audit, and push boundary

**Files:** phase-two runbook and any generated source-port ledger updates.

**Interfaces:** Produce a handoff that identifies the pushed commit, exact unexecuted test commands, and server-only verification still required.

- [x] Document authoritative BSP/RTC source commits, adaptation boundaries, calibration, four modes, and artifact locations.
- [x] Inspect all changed Python for Python-version-incompatible syntax and imports; inspect the complete diff for restored main-only surface.
- [x] Record every test as `NOT RUN — user requires server execution after push`; make no passing claim.
- [x] Request task and whole-branch code review, resolve all load-bearing findings, and ensure the worktree is clean.
- [ ] Push `HEAD:refs/heads/refactor/pi05-libero-bsp-slim`, report the pushed SHA, and stop without server or main operations.
