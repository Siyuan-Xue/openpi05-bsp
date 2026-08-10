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

The public lightweight CI uses one test style and one runner: pytest. It runs Ruff, stdlib-only
repository/core/LIBERO contracts, lightweight training/checkpoint planning tests, documentation/link/deletion-audit
contracts, and the isolated Python 3.8 `openpi-client` suite. “Stdlib-only” describes the code under test; the
pytest runner is the only explicitly requested third-party test tool in that job; uv resolves its small transitive
runtime dependencies. CI deliberately does not download models or data and does not run CUDA, EGL, training, or
rollouts.

The Python 3.11 job uses pytest 9.0.3. The simulator-compatible Python 3.8 client job must temporarily remain on
pytest 8.3.5, because pytest 9 no longer supports Python 3.8; that job directs pytest temporary files to the
job-private GitHub runner directory to contain the affected tmpdir surface documented in
[GHSA-6w46-j5rx-g56g](https://github.com/advisories/GHSA-6w46-j5rx-g56g). Remove this compatibility pin when the
LIBERO client moves to Python 3.10+.

Write tests as native pytest functions or `Test*` classes with plain `assert`, `pytest.raises`, fixtures, and
parametrization. Do not introduce `unittest.TestCase`, `self.assert*`, `subTest`, or `python -m unittest`; a
repository contract and Ruff's pytest-style rules enforce this boundary.

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
