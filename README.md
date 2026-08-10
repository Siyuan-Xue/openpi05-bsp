# OpenPI π0.5 + LIBERO B-spline Policy

## Purpose

This repository is a deletion-oriented fork of
[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi). It keeps one auditable experiment:
compare ordinary action chunks with a B-spline Policy (BSP) action representation while holding the π0.5 base model,
LIBERO data, seed, and evaluation protocol fixed. BSP is adapted to OpenPI; this repository does not claim to reproduce
the paper's tables or assume that BSP must outperform the baseline.

The retained BSP implementation derives from the MIT-licensed
[B-spline Policy reference code](https://github.com/B-spline-policy/bspline-policy). OpenPI attribution and license terms
remain in [LICENSE](LICENSE) and [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt), and implementation attribution remains in
`src/openpi/training/bsp.py`.

## Fixed stack

| Surface | Fixed choice |
|---|---|
| Model and trainer | π0.5 JAX with Flax, Optax, and Orbax on Python 3.11.9 |
| Simulator | Official LIBERO v2.0 on Python 3.8.20 |
| Inference boundary | WebSocket policy server plus the lightweight `openpi-client` |
| Baseline action | Native LIBERO action chunks |
| BSP action | Full-episode cubic spline targets decoded to eight native actions |

The production registry intentionally contains only these five configurations:

- `pi05_libero`
- `pi05_libero_baseline_h16`
- `pi05_libero_bsp_h16`
- `pi05_libero_baseline_lora_h16`
- `pi05_libero_bsp_lora_h16`

The phase-one A/B runs use seed 42, effective batch size 256, no EMA, 10,000 optimizer steps, and fixed checkpoints at
`0 / 1000 / 2000 / 5000 / 10000`. Full-finetuning configurations remain explicit for later hardware validation; the
current experiment uses the two LoRA configurations.

## Shortest run path

1. Follow the [H20 server runbook](docs/pi05_libero_bsp_phase1_server.md). It is the single source for environment,
   asset, training, evaluation, and archival commands.
2. Apply the [normalization gates](docs/norm_stats.md) so baseline and BSP use separate action statistics while their
   state statistics remain equal.
3. Use the [remote-inference boundary](docs/remote_inference.md) between the Python 3.11 JAX server and Python 3.8
   simulator.
4. Run the retained [LIBERO evaluator](examples/libero/README.md), then compare every fixed baseline/BSP checkpoint.

Default pytest is an offline CPU gate. GPU training, checkpoint reload, EGL rendering, and LIBERO rollout require the
server runbook and must not be inferred from local or GitHub CI results.

## Learning path

Start with [Repository architecture](docs/repository_architecture.md). It explains the upstream reading order, the four
BSP integration points, and which removed upstream modules to study when moving beyond this specialized fork.
