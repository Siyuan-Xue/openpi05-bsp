# Repository architecture

This document is a learning map, not an experiment-status record or an operations guide. Use the
[H20 server runbook](pi05_libero_bsp_phase1_server.md) for all server commands and acceptance gates.

## 1. Fork point and slim scope

The authoritative base is Physical Intelligence OpenPI at fork point
[`15a9616`](https://github.com/Physical-Intelligence/openpi/commit/15a9616a00943ada6c20a0f158e3adb39df2ccac).
This fork removes upstream surfaces that do not participate in one π0.5 JAX + official LIBERO v2.0 experiment and adds
one native BSP vertical slice. It is intentionally recognizable as OpenPI rather than a new framework.

The retained boundary is small:

| Layer | Retained surface |
|---|---|
| Model | π0.5 JAX, Gemma, SigLIP, LoRA, and continuous flow-matching actions |
| Data | Official `physical-intelligence/libero@v2.0` through the LeRobot loader |
| Training | The upstream-style config, loader, optimizer, sharding, and Orbax checkpoint flow |
| Policy | LIBERO transforms, JAX policy construction, and WebSocket serving |
| Fork additions | BSP targets and sidecar, isolated A/B normalization, deterministic evaluation, and paired reporting |

The five explicit configurations are `pi05_libero`, `pi05_libero_baseline_h16`, `pi05_libero_bsp_h16`,
`pi05_libero_baseline_lora_h16`, and `pi05_libero_bsp_lora_h16`. Keeping them explicit makes baseline/BSP and full/LoRA
differences visible without introducing another configuration abstraction.

Removed paths and their exact history remain available through Git; duplicating a directory snapshot or per-file deletion
ledger here would turn a learning path into a stale audit artifact.

## 2. Upstream OpenPI main-pipeline reading order

Read the repository in the same direction that data moves. The following order gives each file enough context without
requiring a tour of every implementation detail.

1. `src/openpi/training/config.py` defines `DataConfig`, `TrainConfig`, the five registry entries, model dimensions,
   transforms, assets, optimizer settings, and checkpoint cadence.
2. `src/openpi/models/pi0_config.py` turns the selected model configuration into π0.5, while
   `src/openpi/models/pi0.py` defines the flow-matching loss and sampler. `gemma.py`, `siglip.py`, and `lora.py` supply
   the retained language, vision, and adapter components.
3. `src/openpi/training/data_loader.py` opens the LeRobot dataset and applies the transform groups from the data
   configuration. `src/openpi/transforms.py` contains the reusable repack, normalization, tokenization, and padding
   operations.
4. `scripts/train.py` creates the loader and model state, accumulates micro-batch gradients, performs one optimizer step,
   and delegates checkpoint persistence to the retained training utilities.
5. `src/openpi/policies/policy_config.py` restores a checkpoint and rebuilds the same transforms around an inference
   policy. `src/openpi/policies/libero_policy.py` is the LIBERO-specific observation and action boundary.
6. `scripts/serve_policy.py` exposes that policy through `src/openpi/serving/websocket_policy_server.py`. The simulator
   side stays lightweight in `packages/openpi-client`, while `examples/libero/main.py` drives evaluation and
   `scripts/compare_libero_phase1.py` creates the paired report.

The key invariant is symmetry: training and inference are configured from the same `TrainConfig` and normalization
assets. The WebSocket boundary separates the Python runtimes; it does not define a second model stack.

## 3. BSP integration through the native pipeline

BSP changes the representation of the action target, not the observation path, π0.5 loss, optimizer, checkpoint format,
or evaluator schema. Its four integration points are deliberately direct:

| Stage | Baseline path | BSP path | Invariant |
|---|---|---|---|
| Dataset | LeRobot yields ordinary action sequences | `bsp_dataset.py` fits full episodes and stores targets plus a frame mapping in a sidecar; `BspLeRobotDataset` substitutes the mapped target | Images, state, prompt, episode identity, and sample order are unchanged |
| Normalization | `compute_norm_stats.py` measures native action chunks | The same script measures all BSP parameter rows under a distinct asset id | State statistics must match; action statistics must remain separate |
| Training | The loader passes native actions into the model transform and flow loss | `DataConfig.use_bsp` and `bsp_cache_path` select the verified sidecar before the same transforms and loss | No BSP-specific reconstruction, smoothness, or monotonicity loss is added |
| Policy output | `LiberoOutputs` returns the first seven action channels | `BspLiberoOutputs` runs after unnormalization, keeps the eight parameter channels, and calls `bsp.decode_actions` | Both variants return native seven-dimensional LIBERO actions to the evaluator |

`src/openpi/training/bsp.py` is the sequential algorithm module. It records the B-spline Policy attribution, fixes the
cubic fitting protocol, validates cache identity and shapes, repairs descending knots after unnormalization, and decodes
within the valid knot interval without extrapolation. `src/openpi/training/data_loader.py` contains the only dataset-side
branch: when `use_bsp` is false, the authoritative upstream-style route is unchanged.

At inference, `policy_config.create_trained_policy` orders transforms so quantile unnormalization occurs before the
LIBERO output transform. This ordering is essential because spline knots and control points only have their algorithmic
meaning in the original parameter scale.

## 4. What to revisit in full upstream

This repository is a reading bridge, not a replacement for full OpenPI. After understanding the retained main path,
return to the upstream tree at the fork point for the surfaces below:

| Goal beyond this fork | Upstream modules to revisit | Why they are absent here |
|---|---|---|
| PyTorch training or serving | `src/openpi/models_pytorch/`, `scripts/train_pytorch.py`, and conversion examples | This experiment fixes the JAX backend |
| FAST discrete actions | `pi0_fast.py`, `gemma_fast.py`, and the FSQ tokenizer | BSP and the baseline both use the continuous π0.5 flow head |
| Other robots | ALOHA and DROID policy transforms, examples, and the ALOHA submodule | LIBERO is the only retained observation/action schema |
| RLDS or new data conversion | the DROID RLDS loader and upstream dataset converters | The experiment consumes the official LeRobot LIBERO snapshot |
| Generic robot runtime | `openpi-client` runtime agents and the simple client example | The retained client only needs transport, evaluation, and reporting |
| Containerized deployment | upstream container examples and deployment documentation | The accepted experiment environment follows the host-only runbook |
| Broader examples and notebooks | upstream notebooks, UR5, DROID, and ALOHA examples | They would imply support outside the tested vertical slice |

When restoring any of these capabilities, compare against authoritative upstream rather than copying a deleted file in
isolation: dependencies, configuration registry entries, tests, and deployment assumptions form one feature surface.
The fork deliberately preserves upstream core module names so this comparison remains straightforward.
