# π0.5 + LIBERO B-spline Policy 复现仓库

本仓库是 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)
的专项 fork，只保留 **π0.5 JAX + 官方 LIBERO v2.0 + B-spline Policy
(BSP)** 第一阶段 A/B 复现闭环：同一个 `pi05_base`、同一数据、同一 seed，比较
普通 action chunk 与 BSP 动作表示。论文没有评测 π0.5 或 LIBERO，因此这里是把 BSP
动作表示迁移到 OpenPI，而不是逐项复跑论文表格。

> 实验冻结说明：正在运行或等待验收的服务器实验仍使用 tag
> `phase1-runtime-2c09840`（commit
> `2c098404a3cce0c86f0b863dcd8d3aeb18a55d94`）。瘦身分支
> `refactor/pi05-libero-bsp-slim` 在 baseline/BSP short10k 与全部评测完成前不得部署到
> 该服务器。瘦身分支只声明代码和轻量合同已经实现；GPU、数据、EGL、训练与 rollout
> 必须在 H20 环境重新验收。

## 支持范围

仓库的唯一目标链路是：

```text
official LIBERO v2.0
  -> baseline action chunks / BSP sidecar targets
  -> isolated baseline and BSP norm stats
  -> pi0.5 JAX full or LoRA training
  -> WebSocket policy server
  -> LIBERO four-suite evaluator
  -> paired baseline/BSP report
```

生产配置表只注册以下五项：

| 配置 | 动作协议 | 用途 |
|---|---|---|
| `pi05_libero` | 官方 horizon 10 baseline | 官方 `pi05_libero` checkpoint 的环境校准 |
| `pi05_libero_baseline_h16` | 16 个普通动作 | 未来硬件上的全量微调复验 |
| `pi05_libero_bsp_h16` | 16 × 8 spline 参数，推理解码 8 步 | 未来硬件上的 BSP 全量微调复验 |
| `pi05_libero_baseline_lora_h16` | 16 个普通动作 | 当前第一阶段 baseline |
| `pi05_libero_bsp_lora_h16` | 16 × 8 spline 参数，推理解码 8 步 | 当前第一阶段 BSP |

当前第一阶段采用 LoRA、seed 42、有效 batch 256、命令行
`micro_batch_size=64`、不使用 EMA、10,000 optimizer steps。固定验收里程碑是：

```text
0 / 1000 / 2000 / 5000 / 10000
```

0 是训练前但已装载 `pi05_base` 的可审计 checkpoint；其余数字都是 optimizer step，
不是 micro-step。两组各五个 checkpoint 必须逐点评测，不选择“最佳 checkpoint”。全量配置
继续保留，但当前单张 H20 不承担正式 full-finetune 验收。

## 从哪里开始

- [服务器第一阶段 runbook](docs/pi05_libero_bsp_phase1_server.md)：阿里云 DSW
  双 Python 环境、数据、sidecar、norm、训练、推理、评测与报告的唯一操作顺序。
- [仓库架构与删除审计](docs/repository_architecture.md)：瘦身前后目录、组件关系、
  删除代价与追溯方式。
- [Normalization 统计门禁](docs/norm_stats.md)：baseline/BSP 独立资产与
  state-equality/action-separation 验收。
- [LIBERO WebSocket 推理](docs/remote_inference.md)：Python 3.11 policy server 与
  Python 3.8 simulator client 的接口。
- [LIBERO evaluator 快速说明](examples/libero/README.md)：冒烟、官方校准和单次评测入口。

服务器按 runbook 建立锁定环境；不要把本地开发机的 Python 或依赖状态当成服务器验收证据。
关键 CLI 如下：

```bash
# 官方数据与 BSP sidecar
python scripts/prepare_libero_bsp.py --help

# 两套独立 normalization 统计
python scripts/compute_norm_stats.py --help

# JAX 训练
python scripts/train.py pi05_libero_baseline_lora_h16 --help
python scripts/train.py pi05_libero_bsp_lora_h16 --help

# policy server、LIBERO evaluator 与最终比较
python scripts/serve_policy.py --help
python examples/libero/main.py --help
PYTHONPATH=packages/openpi-client/src python scripts/compare_libero_phase1.py --help
```

## BSP 固定协议

BSP 以论文为算法语义，以作者 MIT 代码补足 FITPACK 实现细节：

- 在每个完整 episode 的原始 7D delta-action（含 gripper）上独立拟合；
- cubic degree 3、chunk size 10、stride 1、frame-index 时间轴；
- 最大绝对误差 0.002、FITPACK smoothing `1e-12`；
- 每个 target 为 16 行，通道 `0:7` 是控制点，通道 `7` 是 knot；
- sidecar 缓存 episode-start knot，取样时转换为当前 episode-local frame 相对 knot；
- 16 knots 对应 12 个有效 control points；尾部按最后合法值填充；
- 反归一化后修复下降 knot，再在 `[knots[3], knots[-4]]` 内等距解码 8 步，禁止外推。

训练不增加重建、平滑或单调性额外 loss。baseline 执行预测序列前 8 步，BSP 执行解码
出的 8 步；两者都在原生 10 Hz 下每 8 步重规划。第一阶段不启用时间缩放、异步执行、
segment alignment 或 gripper 阈值化。

## 主要代码入口

| 路径 | 职责 |
|---|---|
| `src/openpi/models/` | π0.5 JAX、Gemma、SigLIP、LoRA |
| `src/openpi/training/bsp.py` | spline target、sidecar 格式、解码与 MIT attribution |
| `src/openpi/training/bsp_dataset.py` | 官方 LeRobot LIBERO episode 映射和缓存身份 |
| `src/openpi/training/config.py` | 五个配置与固定 short10k 里程碑 |
| `src/openpi/training/data_loader.py` | baseline/BSP 的 LeRobot 数据装配 |
| `scripts/train.py` | 梯度累积、optimizer step、Orbax checkpoint |
| `src/openpi/policies/libero_policy.py` | LIBERO 输入与 baseline/BSP 输出 transform |
| `scripts/serve_policy.py` | JAX checkpoint WebSocket 服务 |
| `examples/libero/main.py` | 四套件 rollout 与逐 episode 审计产物 |
| `packages/openpi-client/` | 轻量 WebSocket client、评测记录与 paired report |

`openpi-client` 不是另一套模型。它是可安装到 LIBERO Python 3.8 环境中的轻量通信包，
把 observation 通过 WebSocket 发给 Python 3.11/JAX 服务，再接收 action；同时承载确定性
seed、错误分类、manifest、JSONL/CSV/JSON 产物和五个里程碑的配对报告。

## 不再支持的上游能力

这不是通用 OpenPI 发行版。瘦身分支不再承诺 FAST/FSQ、PyTorch 模型或训练、ALOHA、
DROID、UR5、RLDS 转换、通用机器人 runtime、notebook 示例以及 Docker/Compose 部署兼容性。
删除这些入口只对本仓库的目标闭环“无影响”；如果要恢复其他机器人、模型后端或容器部署，
请从 tag `phase1-pre-slim-1b976fc` 查看清理前实现，或直接使用上游 OpenPI。

## 测试边界

公开 GitHub CI 只运行 CPU 轻量门禁：Ruff、依赖/配置/目录/文档合同，以及独立
`openpi-client` 测试。pre-commit 在创建 commit 前做同类代码卫生检查。它们不会下载模型
或数据，也不会证明 CUDA、EGL、JAX 训练和 LIBERO rollout 可用；完整门禁只能按服务器
runbook 在 H20 环境执行。

## 来源与许可证

- OpenPI 模型、训练和服务骨架来自
  [Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，许可证见
  [LICENSE](LICENSE) 与 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
- BSP 论文语义和 spline 实现参考
  [B-spline-policy/bspline-policy](https://github.com/B-spline-policy/bspline-policy)。移植代码的
  MIT attribution 保留在 `src/openpi/training/bsp.py`。
- LIBERO 通过锁定的 `third_party/libero` gitlink 使用，其上游许可证随子模块保留。

本仓库只声明可审计的复现流程，不预设 BSP 必须优于 baseline；完整、可配对的负结果同样
是有效结果。
