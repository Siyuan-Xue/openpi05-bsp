# Contributing to the π0.5 + LIBERO BSP fork

This repository is a focused reproduction fork, not a general OpenPI distribution. Contributions should preserve
the audited closure documented in [the architecture guide](docs/repository_architecture.md): official LIBERO v2.0,
π0.5 JAX, baseline/BSP full and LoRA training, WebSocket evaluation, and paired phase-one reporting.

## In scope

- correctness fixes for BSP fitting, sidecar identity, normalization, decoding, or LIBERO transforms;
- deterministic and auditable training/checkpoint/evaluation behavior;
- the five retained training configurations;
- the Python 3.11 policy server / Python 3.8 LIBERO client boundary;
- lightweight contracts, documentation, and reproducibility metadata for this experiment.

Support for other robots, FAST/FSQ, PyTorch model execution, RLDS conversion, generic robot runtimes, notebooks,
or container orchestration is intentionally out of scope. Propose such work to the appropriate upstream project or
maintain it on a separate branch; do not silently re-expand the production registry or dependency surface here.

## Before changing code

1. Work on a topic branch or isolated worktree; do not commit experimental refactors directly to `main`.
2. State which phase-one invariant changes and why. Protocol changes require explicit review before implementation.
3. Add a focused regression/contract test first and observe the expected failure.
4. Never include access tokens, machine identities, user data, model weights, datasets, checkpoints, videos, or
   experiment logs in a commit.

The frozen runtime tag `phase1-runtime-2c09840` identifies active experiments. A development branch must not be
deployed into those runs or used to rewrite their manifests.

## Local and CI checks

Install hooks in an already prepared development environment:

```bash
pre-commit install
pre-commit run --all-files
```

The public lightweight CI runs Ruff, 23 dependency-free repository/core/LIBERO contracts, documentation/link/
deletion-audit contracts, and the isolated Python 3.8 `openpi-client` suite. It deliberately does not download
models or data and does not run CUDA, EGL, training, or rollouts.

When dependencies change, regenerate `uv.lock` with the pinned uv workflow and explain direct dependency additions.
Resolver-required transitive packages may remain; an unused package is not justification for hand-editing the lock.

## Server-only evidence

Changes affecting data, model shapes, optimizer/checkpoint behavior, policy output, or evaluator semantics also need
the applicable gates from the [server runbook](docs/pi05_libero_bsp_phase1_server.md):

- CPython 3.11.9 frozen sync, SciPy 1.15.3, and JAX GPU discovery;
- CPython 3.8.20 LIBERO client and EGL reset/render/step;
- sidecar build/full verify and baseline/BSP norm gates;
- finite pilot loss/grad/parameter norms and exact optimizer steps;
- checkpoint assets and four-suite paired evaluation contracts.

Do not claim these passed from a laptop or GitHub CPU runner. Record command, code SHA, input identity, exit status,
and artifact hashes from the H20 server.

## Pull request content

A reviewable change includes:

- a concise problem statement and target-closure impact;
- RED/GREEN test evidence and the full verification commands used;
- a deletion or dependency audit when the supported surface changes;
- documentation updates for changed CLI, paths, manifests, or protocol;
- explicit remaining server-only gates and known limitations.

Keep large artifacts outside Git. On the current server, only `/mnt/data/siyuanxue` is an approved writable data
namespace; repository examples must not direct writes elsewhere under `/mnt/data`.

Contributions remain subject to [LICENSE](LICENSE) and [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt). BSP-derived code must
retain the MIT attribution embedded in `src/openpi/training/bsp.py`.
