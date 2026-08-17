# LIBERO evaluator

This is the repository's only simulator example. It runs the locked
[`third_party/libero`](../../third_party/libero) revision under Python 3.8.20 and sends observations to the Python 3.11.9
JAX policy server through `openpi-client`.

All environment setup, asset paths, policy-server invocations, evaluator commands, hashes, and H20 acceptance checks
live in the [canonical server runbook](../../docs/pi05_libero_bsp_phase1_server.md). Keeping those commands in one place
prevents this example from drifting into a second operations guide.

## Retained files

| File | Role |
|---|---|
| `main.py` | Four-suite evaluator, deterministic seeds, retry classification, and per-episode artifacts |
| `requirements.in` | Direct simulator dependencies |
| `requirements.txt` | Locked Python 3.8 installation input |

The simulator and OpenPI server remain separate environments. The evaluator sends images, state, prompt, and an
inference seed; it receives native LIBERO actions. Baseline h16 executes the first eight predicted actions, while BSP
receives eight actions already decoded by the server.

## Timing and schema v3

The evaluator keeps four independent clocks: the LeRobot dataset is indexed at
10 FPS, source demonstrations identify a 20 Hz environment, evaluation
dynamics run at exactly 20 Hz, and selected MP4 files default to 40 FPS.
Dataset and source rates are provenance only. Video synthesis uses control
steps, the evaluation control rate, the selected video FPS, and measured
control stalls.

Use `--args.control-freq 20 --args.video-fps 40`. The optional
`--args.video-show-inference-waits` changes only selected videos and their
audits; it adds no environment step, dummy action, or sleep. Request latency
and control stall are separate events, and only included stalls freeze video.
The evaluator resolves `manifest.code_sha` from a clean checkout and fails
closed if Git identity cannot be established.

## Audit outputs

Each evaluation run records `manifest.json`, `episodes.jsonl`, `tasks.csv`, `suites.csv`, and `summary.json`, with an
artifact-error log only when needed, sampled success/failure videos, and
`video_audit.jsonl` for selected videos. Zero-step policy failures retain a
request-time source frame so their failure video remains auditable. Missing
stall sources, missing overlays, encoder/readback failures, or mismatched
audit records make the run incomplete.

The paired reporter compares baseline and BSP only at `0 / 1000 / 2000 / 5000 / 10000`. It reads variant, checkpoint,
dataset, code, cache, normalization, runtime, and seed identity from each manifest rather than inferring identity from
directory names. Formal comparisons accept schema v3 only; schema-v2 outputs
remain immutable archive inputs and are never mixed or auto-upgraded.
