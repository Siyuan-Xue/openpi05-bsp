# π0.5 + LIBERO BSP 第一阶段 0k/5k 验收扩展设计

日期：2026-08-09  
状态：已确认方案 A，等待书面规格复核

## 1. 目标与边界

第一阶段的固定验收里程碑由 `10k / 20k / 30k` 扩展为：

```text
0k / 5k / 10k / 20k / 30k
```

Baseline 和 BSP 各评测五个 checkpoint，因此固定比较从 6 个运行扩展为 10 个运行。
训练仍执行 30,000 个 optimizer steps；本变更不修改模型、数据集、学习率、有效 batch、
micro-batch、seed、loss、动作表示、解码协议或评测 episode 数。

`0k` 的定义必须是：模型已经按对应实验配置构造，并从同一个 `pi05_base` 加载权重，
但尚未执行任何 optimizer update。它不是官方 `pi05_libero` checkpoint，也不是第一个训练后
checkpoint 的别名。

## 2. 固定里程碑

代码中使用唯一的有序里程碑定义：

```python
(0, 5_000, 10_000, 20_000, 30_000)
```

四个第一阶段训练配置都采用同一组里程碑：

- `pi05_libero_baseline_h16`
- `pi05_libero_bsp_h16`
- `pi05_libero_baseline_lora_h16`
- `pi05_libero_bsp_lora_h16`

全量微调和 LoRA 是两个可选实验族。一次 A/B 比较必须全部来自同一实验族，禁止将全量
Baseline 与 LoRA BSP（或反向组合）混入同一报告。

## 3. Checkpoint 保存与保留

### 3.1 显式永久保留步骤

`TrainConfig` 新增显式永久保留步骤字段。第一阶段四个配置固定为
`(0, 5_000, 10_000, 20_000, 30_000)`。

继续每 1,000 optimizer steps 保存恢复 checkpoint。checkpoint manager 只保留：

- 最近一个非里程碑恢复点；
- 上述五个永久里程碑。

不能仅把 `keep_period` 改为 5,000，因为那会额外永久保留 15k 和 25k，显著增加磁盘占用。
实现应使用当前锁定 Orbax 版本支持的精确保留谓词；没有显式步骤的其他 OpenPI 配置继续
沿用原有 `keep_period` 行为。

### 3.2 Step 0 写入时机

对于包含永久步骤 `0` 的新训练：

1. 创建数据加载器并加载对应的 norm assets；
2. 构造训练状态并加载 `pi05_base` 权重；
3. 在编译或执行任何梯度更新前调用现有 checkpoint 保存通路，写入 `0/`；
4. 等待异步保存完成；
5. 从 optimizer step 1 开始训练。

因此 step 0 checkpoint 与训练 checkpoint 具有相同结构，至少包含：

- `params/`：可用于推理的参数；
- `train_state/`：可恢复的训练状态；
- `assets/<asset_id>/norm_stats.json`：该实验动作协议对应的归一化统计。

Baseline 与 BSP 的 step 0 权重初始化相同，但 norm assets、输出语义和推理解码路径不同。
这种差异是实验定义的一部分，而不是权重差异。

### 3.3 恢复语义

目录中只有 step 0 时，`--resume` 必须恢复 step 0，而不能把它视为“没有 checkpoint”。
恢复后从 step 1 开始，且不得重复覆盖 step 0。目录完全没有 checkpoint 时才回退到全新初始化。

Checkpoint 只能在 optimizer-step 边界落盘；梯度累积的中间 micro-batch 仍不能保存或恢复。

## 4. 评测与配对协议

每个 checkpoint 继续完成四套件 × 10 tasks × 50 rollouts，即每个 checkpoint 2,000 回合。
十个 checkpoint 的固定全量验收共 20,000 回合。

每个里程碑都形成一对：

```text
(baseline, step) <-> (bsp, step)
```

包括 step 0 在内，A/B 必须使用一致的 LIBERO initial state 和确定性 flow-noise seed 派生规则。
Baseline 按普通 action chunk 执行；BSP 按既定 spline 解码执行。step 0 不获得任何特殊推理逻辑。

评测 manifest 中的 checkpoint step 必须与 checkpoint 路径末级目录一致，并允许合法值 `0`。
同一报告的十个 checkpoint 身份必须全局唯一。

## 5. 报告器扩展

固定比较报告接受且只接受十个运行目录，目录次序不具有语义；报告器继续从 manifest 内容识别
variant、step 和实验族。

报告器必须验证：

- 五个里程碑的 Baseline/BSP 运行各恰好一个；
- 所有运行属于同一训练族（全量或 LoRA）；
- 数据 revision、cache fingerprint、norm hash、初始状态、评测 seed 和协议字段一致；
- BSP 运行使用 BSP cache/decoder，Baseline 运行不使用；
- 10 个运行均完成基础设施和产物门禁。

输出学习曲线和逐里程碑比较时按 `0, 5k, 10k, 20k, 30k` 排序。总体报告的
`milestones`、总运行数和总 episode 数同步更新；命令行帮助和成功消息不得继续写“six”或
“10k/20k/30k only”。

## 6. 兼容性与错误处理

- 不改变非第一阶段训练配置的 checkpoint 保留规则。
- 旧的 6-run 报告输入必须明确失败，并说明现在要求 10 个运行，而不是静默生成不完整报告。
- 缺失 step 0 或 step 5k 时，报告必须在生成任何最终产物前失败。
- 混合全量/LoRA 配置、重复 checkpoint、step 与路径不一致、错误 norm/cache hash 均必须失败。
- checkpoint 初始化保存失败时不得开始 optimizer step 1。
- 不使用官方 `pi05_libero` 作为 step 0 的替代品。

## 7. 测试策略

实施遵循 TDD，至少覆盖：

1. 第一阶段配置暴露五个精确永久里程碑，且不永久保留 15k/25k。
2. 全新训练在第一个 optimizer update 前保存 step 0。
3. 只有 step 0 的训练目录能够从 step 0 恢复并从 step 1 继续。
4. 1k 恢复点仍按原间隔保存，5k/10k/20k/30k 被永久保留。
5. 报告 CLI 要求恰好 10 个运行目录。
6. 报告接受完整的全量十运行或 LoRA 十运行，并拒绝混合实验族。
7. 报告拒绝旧六运行、缺失 0k/5k、重复身份及 step/path 不一致。
8. 固定报告产物包含五个里程碑、十个运行和 20,000 个 episode 的协议元数据。
9. 既有 checkpoint 边界、梯度累积、BSP 数据/解码和评测测试继续通过。

服务器正式训练开始前还要执行轻量合同测试，并确认新训练目录不存在、GPU 空闲、主盘容量
满足门禁。正式训练开始后首先验证两个 step 0 checkpoint 的结构和 manifest 身份，再让训练
继续推进。

## 8. 验收结果解释

0k 用于观察“同一 base 权重经过两种动作表示和归一化/解码协议后”的初始表现；5k 用于补充
早期学习阶段；10k/20k/30k 保持原协议。BSP 不被预设为必须优于 Baseline，所有五个里程碑
均按固定顺序报告，不挑选最好 checkpoint。
