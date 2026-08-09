# π0.5 + LIBERO BSP 第一阶段 10k 短周期协议设计

日期：2026-08-09
状态：已确认方案 A，等待书面规格复核

## 1. 决策与目标

第一阶段不再训练到 30,000 optimizer steps。Baseline 与 BSP 统一训练到 10,000
optimizer steps，并在以下五个固定里程碑验收：

```text
0k / 1k / 2k / 5k / 10k
```

本设计取代先前 `0k / 5k / 10k / 20k / 30k` 的第一阶段里程碑。缩短训练周期不会改变
模型架构、LIBERO v2.0 数据、BSP target、norm stats、学习率、优化器、有效 batch 256、
micro-batch 64、seed 42、loss、动作解码或每个 checkpoint 的评测规模。

按当前 H20 实测约 29.6 秒/optimizer step 估算，每个 variant 的 10k 训练约需 3.4 天，
Baseline 与 BSP 串行约需 6.9 天。该估算只用于资源规划，不是验收条件。

## 2. 训练配置

以下四个第一阶段配置使用完全相同的训练上限和永久里程碑：

- `pi05_libero_baseline_h16`
- `pi05_libero_bsp_h16`
- `pi05_libero_baseline_lora_h16`
- `pi05_libero_bsp_lora_h16`

固定配置为：

```python
num_train_steps = 10_000
save_interval = 1_000
keep_period = 10_000
permanent_checkpoint_steps = (0, 1_000, 2_000, 5_000, 10_000)
```

`save_interval=1_000` 仍提供整数千步恢复点。永久保留谓词确保上述五个验收点不会被清理；
非里程碑恢复点沿用现有 Orbax 保留策略。checkpoint 只能在完整 optimizer-step 边界保存，
不得在梯度累积的 micro-step 中间落盘。

LoRA 正式路线继续使用有效 batch 256、micro-batch 64、无 EMA 和 seed 42。全量微调配置
保留在仓库中作为独立可选实验族，但不得与 LoRA 结果混入同一个 A/B 报告。

## 3. 当前 30k 运行的终止与审计

正在运行的旧实验 `phase1-seed42-baseline` 属于已废止的 30k 协议。实施时必须：

1. 从 PID 文件读取进程并核对命令确为该训练；
2. 使用 `SIGTERM` 请求正常终止，不使用 `SIGKILL`；
3. 等待进程退出并确认 GPU 释放；
4. 保留已有 step 0、日志、PID 文件和 W&B offline 记录；
5. 写入独立审计记录，注明它因协议变更而终止，而非 OOM、NaN 或训练故障；
6. 不删除、覆盖、改名或把旧目录用于恢复新协议。

新实验采用唯一名称：

```text
phase1-short10k-seed42-baseline
phase1-short10k-seed42-bsp
```

两个新实验都必须从同一个 `pi05_base` 和各自正式 norm assets 开始，不从旧 30k 运行恢复。

## 4. 评测与报告

每个 checkpoint 仍执行四套件 × 10 tasks × 50 rollouts，即 2,000 回合。Baseline 与 BSP
各有五个 checkpoint，因此第一阶段固定为 10 个评测运行、20,000 回合。

报告器接受且只接受以下配对：

```text
(baseline, 0)     <-> (bsp, 0)
(baseline, 1000)  <-> (bsp, 1000)
(baseline, 2000)  <-> (bsp, 2000)
(baseline, 5000)  <-> (bsp, 5000)
(baseline, 10000) <-> (bsp, 10000)
```

必须继续验证训练族、数据 revision、sidecar fingerprint、norm hash、评测 seed、initial state、
checkpoint 路径和 manifest step。旧的 30k 五里程碑输入必须明确失败，不能被静默解释为新协议。

总体 episode 数保持 20,000，但学习曲线的横轴和统计配对改为 `0k/1k/2k/5k/10k`。
不得从五个里程碑中挑选“最佳 checkpoint”。

## 5. 服务器运行与持久化

训练继续把 Orbax checkpoint 写入 `/root/openpi-bsp-work/experiments/checkpoints` 的本地
overlay，不直接写入 `ossfs2`。永久里程碑完成后，使用单文件归档、校验和与原子发布流程
复制到 `/mnt/data/siyuanxue/openpi-bsp/experiments/checkpoint-archives/`；不得写入
`/mnt/data` 下其他目录。

正式启动前必须验证：

- 服务器代码 SHA 与远端 `main` 一致且工作区干净；
- baseline/BSP norm、LIBERO 数据和 BSP sidecar 身份未变化；
- GPU 没有其他 compute process；
- 新 checkpoint、日志和 PID 路径不存在；
- `/root` 可用空间满足 step 0 和后续整数千步 checkpoint 的瞬时写入；
- 四个配置都声明相同的 10k 上限和五个永久里程碑。

新 baseline 启动后必须先验收 `0/params`、`0/train_state` 和 `0/assets`，且日志无
OOM、`RESOURCE_EXHAUSTED`、Traceback、NaN 或 Inf，再继续训练。

## 6. 错误处理与恢复

- 旧 30k 运行终止失败时停止，不升级到 `SIGKILL`。
- 新路径发生碰撞时停止，不使用 `--overwrite`。
- 新训练异常退出时保留全部证据，不自动重试或降低参数。
- 恢复必须来自同一短周期实验的完整 optimizer-step checkpoint，并显式使用 `--resume`。
- 10k 完成后不得继续到 20k 或 30k；进一步训练属于新的实验协议。
- Baseline 未完成 10k 全部门禁前，不启动 BSP 正式训练。

## 7. 测试策略

实施遵循 TDD，至少覆盖：

1. 四个第一阶段配置的 `num_train_steps` 都精确为 10,000。
2. 四个配置的永久里程碑都精确为 `(0, 1_000, 2_000, 5_000, 10_000)`。
3. step 0 仍在第一次 optimizer update 前保存，并可从 step 0 恢复。
4. 1k 与 2k 被永久保留，10k 是训练终点；20k/30k 不再属于第一阶段协议。
5. 报告器接受完整的十运行新协议，并拒绝旧里程碑、缺失配对、重复身份和混合训练族。
6. 报告协议仍记录 10 个运行和 20,000 个 episode。
7. Docker 与 host evaluator 的帮助、路径和示例命令统一使用五个新里程碑。
8. 现有 BSP 数据、norm、梯度累积、checkpoint 边界和评测测试继续通过。

## 8. 验收定义

代码验收要求配置、报告器、合同测试和服务器 runbook 对新协议只有一个一致定义。运行验收要求
旧实验已保留证据并正常终止，新 baseline 使用新名称启动，step 0 完整落盘且训练进程稳定。

第一阶段的最终实验结果来自 Baseline/BSP 在 `0k/1k/2k/5k/10k` 的固定 A/B 比较。BSP
不被预设为必须优于 Baseline；负结果仍是有效、可审计的复现结果。
