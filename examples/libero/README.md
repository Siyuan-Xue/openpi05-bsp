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

## Audit outputs

Each evaluation run records `manifest.json`, `episodes.jsonl`, `tasks.csv`, `suites.csv`, and `summary.json`, with an
artifact-error log only when needed and sampled success/failure videos for inspection. Infrastructure failures remain
separate from policy failures and make the run incomplete.

The paired reporter compares baseline and BSP only at `0 / 1000 / 2000 / 5000 / 10000`. It reads variant, checkpoint,
dataset, code, cache, normalization, runtime, and seed identity from each manifest rather than inferring identity from
directory names.
