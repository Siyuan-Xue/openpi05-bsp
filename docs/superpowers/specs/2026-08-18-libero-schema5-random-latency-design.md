# LIBERO schema-v5 配对随机延迟异步实验设计

日期：2026-08-18  
基线：`main@3861651ca0f63895926c491dd21dda2194e643a7`  
保护 tag：`pre-random-latency-schema4-3861651`

## 1. 目标与结论边界

本轮实验研究在相同随机推理延迟下，普通 action-chunk 异步、RTC 平滑异步和
BSP 连续样条异步的延迟隐藏、动作边界连续性及吞吐差异。

正式实验只包含三个 10K checkpoint 条件：

1. `baseline_async`；
2. `baseline_rtc`；
3. `bsp_spline_async`。

三组都使用均值 300 ms、标准差 60 ms 的配对正态目标延迟。正式实验不包含
target=0、同步 Baseline、同步 BSP，也不再执行固定 100/200/300/850 ms 容量扫描。
此前 schema-v4 固定 100 ms 容量结果只作为废弃试跑证据保留，不进入 schema-v5
报告。

该注入模拟“推理结果更晚对调度器可见”，用于研究延迟隐藏和吞吐。它不模拟低算力
设备的功耗、显存、多请求并发能力或其他硬件特性。LIBERO underflow 时会暂停仿真，
因此结论不直接等同于真机抽搐或物理成功率改善。

## 2. 不变项

本轮不修改：

- checkpoint、训练参数、模型权重和 norm stats；
- observation 预处理、模型输出、采样步数和 BSP 解码数学；
- LIBERO 动力学、20 Hz 控制频率、episode 最大步数、done 或成功判定；
- 训练 loader、BSP sidecar 内容、CI、依赖和仓库瘦身范围。

视频保持 40 FPS。录像和审计开关不得改变 action、`env.step()` 次数、done、status
或 success。不需要重新训练或重新拟合 BSP sidecar。

## 3. 执行模式

### 3.1 共享的 Baseline 异步调度

`baseline_async` 和 `baseline_rtc` 共用同一 calibration、理论延迟预算、请求启动条件、
请求时刻、容量边界和 underflow 处理。共同启动条件为：

```text
cursor >= max(MIN_START, forecast_delay)
cursor + forecast_delay <= 16
MIN_START = 8
```

在本轮 400 ms 调度预算下，`forecast_delay=8` 个 50 ms 控制周期，因此通常表现为
cursor 8 发起请求。这是两种模式共享的 RTC 调度规则，不是给普通 async 另行硬编码的
实验条件。

初始请求保持同步，以获得第一段动作。后续请求在后台进行；请求未完成时继续执行旧
chunk。旧 chunk 耗尽而结果仍不可用时，调度器记录真实 action underflow 和 control
stall，然后等待结果。

### 3.2 `baseline_async`

普通 async 请求不携带 RTC 的 `previous_model_actions`、`s` 或 `d` 连续性引导。
返回值仍是普通 H16 action chunk。

设请求在旧 chunk cursor `s` 发出，响应可见时已经执行了 `q` 个控制步。安装新 chunk
时跳过其前 `q` 个 action，并从时间对齐的位置继续执行。这样不会把 action 0 延迟到
错误的物理时刻，也不会把 observation 时刻改成另一个实验变量。由于没有 RTC 引导，
旧 segment 最后一个 action 和新 segment 首个实际 action 之间允许出现真实跳变。

### 3.3 `baseline_rtc`

RTC 与普通 async 在相同时间发请求，但携带旧 chunk 的 normalized model actions 及
`s/d` 引导信息。响应安装时使用相同的实际经过步数对齐，由模型生成连续过渡。

因此 `baseline_async` 与 `baseline_rtc` 的主要可控差异是 RTC 连续性引导，而不是
prefetch 时刻、延迟样本或时间对齐策略。

### 3.4 `bsp_spline_async`

BSP async 使用同一 400 ms 理论预算提前请求下一段曲线。结果及时可用时立即安装，
推理 latency 不产生视频冻结；曲线耗尽但结果未到时才记录真实 underflow 和 stall。
BSP 的 sidecar、曲线拟合、degree、origin rate、speedup 和解码保持不变。

## 4. 配对正态延迟

### 4.1 分布

每次请求的目标可见延迟来自：

```text
Normal(mean=300 ms, stddev=60 ms)
```

使用版本化的确定性 `SHA-256 + Box-Muller` 采样器。正式 episode 的采样键包含：

```text
eval_seed / suite / task_id / trial_index / request_ordinal / namespace
```

模式名不进入正式 episode 的采样键，所以三个模式在相同 episode、相同 request 序号
获得相同目标样本。Calibration 使用独立 namespace，不消耗正式 episode 序列。

采样器不依赖 Python `random`、NumPy RNG 或进程启动顺序。正态分布得到负值时使用增加
resample counter 的新哈希重新采样。最终纳秒数使用文档化、跨 Python 版本一致的舍入
规则。manifest 固定记录 sampler 名称、版本、seed、分布参数和负值策略。

### 4.2 注入位置与事件字段

延迟注入发生在 WebSocket response 已返回、single-owner worker 将 outcome 发布给调度器
之前。所有三个模式经过同一个注入点。

每个请求记录：

```text
sampled_target_latency_ns
raw_inference_latency_ns
synthetic_delay_ns
effective_inference_latency_ns
```

关系为：

```text
effective = max(raw, sampled_target)
synthetic_delay = effective - raw
```

等待与时间戳使用 monotonic clock。真实请求超过目标样本时绝不截短。

### 4.3 Calibration 与调度预算

每种模式执行 5 次 warmup 和 20 次 measurement。Warmup 排除连接、JAX 编译和 GPU
首次初始化；measurement 同时记录 raw、sampled、synthetic 和 effective latency。

本轮调度预算不使用仅 20 个随机样本的 empirical p95。配置分布的理论单侧 p95 为：

```text
300 ms + 1.644853626951 × 60 ms = 398.691217617 ms
```

向上对齐 50 ms 控制周期后，三种模式统一使用 400 ms 预算。Empirical calibration
p95 仍写入审计；它偶然超过 400 ms 时只告警，不拒绝运行。正式 episode 中超过
400 ms 的随机长尾不截断，按真实 underflow/stall 记录。

## 5. 动作边界跳变审计

每次新计划安装后，在其第一个 action 真正交给 `env.step()` 时，将它与上一个已执行
action 比较。事件记录：

- activation 所在 control step；
- 前 6 个机械臂 action 维度的 L2 jump；
- 前 6 维的最大绝对 jump；
- gripper action 的绝对 jump；
- execution mode 和 plan activation identity。

三种模式使用同一计算函数和 action 空间。指标只观察已经执行的 action，不修改动作、
不平滑动作，也不为录像额外执行 action。报告使用这些事件定量验证普通 async 的 segment
跳变及 RTC/BSP 的连续性表现。

## 6. 视频

视频固定 20 Hz 控制、40 FPS 编码，并采用流式写入。每个真实 `env.step()` 对应 50 ms
仿真时间；正常画面重复两帧。只有真实 control stall 才增加冻结帧，不使用十倍展示、
dummy action、额外 `env.step()` 或动作重放。

每个视频从首帧开始持续显示：

```text
Cumulative inference wait: X.XX s
```

文字为白色、细暗色描边、无黑色矩形背景。stall 期间累计值按 40 FPS 增长，正常运动
期间保持，下一次 stall 继续累加。异步 latency 非零但未造成 underflow 时不冻结。

视频审计继续验证帧数、40 FPS、期望时长和一帧容差。录像开关不得改变 action 或结果。

## 7. schema-v5 与报告

当前 schema-v4 评测栈原位演进并重命名为 `*_v5`；不同时维护两套重复实现。旧 v4 代码
和结果由 commit `3861651` 及保护 tag 保存。schema-v5 正式报告拒绝 v4、固定延迟或其他
模式输入。

Manifest 新增或冻结：

- `latency_distribution`；
- `scheduling_latency_budget_ns=400000000`；
- empirical calibration 的 raw/effective measurements 和 p95；
- `baseline_async` 的执行参数与 policy protocol；
- sampler 和 action-boundary-jump schema 版本。

正式报告必须恰好接收三个 2000-episode run，且 code SHA、checkpoint step、dataset、seed、
分布身份、控制/视频频率、suite/task/trial 协议一致。主要比较为：

- `baseline_async` 对 `baseline_rtc`：成功率、boundary jump、underflow、stall 和吞吐；
- `bsp_spline_async` 对两种 Baseline：成功率、jump、延迟隐藏比例和吞吐；
- 三组的实际 sampled/raw/effective 分布、控制步数、episode wall time 和 episodes/min。

配置期望 300 ms 与实际样本统计必须分开报告。实际统计包括 mean、stddev、p50、p95 和
max。

## 8. 代码边界

预计只修改：

- `packages/openpi-client` 中的 single-owner worker、LIBERO controller/eval/video/report；
- `examples/libero` 的 schema-v5 evaluator 入口；
- LIBERO policy capability metadata；
- schema-v5 目标测试与运行文档。

不引入仓库瘦身删除、loader resume、CI/依赖重构或其他机器人路径修改。

## 9. Git 与服务器发布

1. 保护 tag 固定 `3861651`；
2. 在本地 `main` 上用普通提交记录设计和实现，不 rebase、不 force-push；
3. 候选 SHA 先推送临时验证引用；
4. 服务器在新的独立 clean worktree 检出同一 SHA，不覆盖现有目录；
5. 门禁全部通过后，才把相同提交 fast-forward 推送到 `origin/main`；
6. 失败使用后续修复或 revert commit，不改写历史。

BSP diagnostics 使用新 clean SHA 对现有 sidecar 做非破坏性 verify，写入新的无碰撞路径；
不重新拟合、不覆盖旧 diagnostics。

## 10. 测试与门禁

本地 TDD 覆盖：

- 正态采样确定性、跨模式配对、namespace 隔离、负值重采样和精确舍入；
- raw 超过 sampled target 时不截短；
- 三模式共享同一 worker 注入点；
- theoretical p95 与 400 ms 预算，empirical p95 只告警；
- 普通 async 与 RTC 相同请求时刻和容量逻辑；
- 普通 async 不发送 RTC 连续性引导，并按实际延迟跳过新 chunk 前缀；
- underflow 只记录真实等待；
- boundary jump 指标来自实际相邻 action；
- 单行累计文字、无黑框、1倍时长和流式编码；
- 视频开关不改变 action、steps、done、status 或 success；
- report 只接受三个匹配的 schema-v5 输入。

服务器门禁依次为：

1. clean SHA、Python环境、LIBERO子模块和policy capability；
2. 完整CPU测试与Python3.8导入；
3. EGL reset/render/step；
4. 三种模式各一个小型随机延迟 smoke；
5. 相同输入下录像开/关的动作与结果一致；
6. MP4、累计文字、帧数、时长、jump和audit回读；
7. 新 SHA 的 BSP sidecar 非破坏性 verify。

门禁完成后停止并向用户提交证据。不得自动启动小样本扫描或三个正式 2000-episode run；
正式验收必须再次获得用户明确授权。

## 11. 最终实验规模

在后续单独授权后，正式实验串行运行：

```text
baseline_async       2000 episodes
baseline_rtc         2000 episodes
bsp_spline_async     2000 episodes
```

一次最多一个 policy server 和一个 evaluator。三组完成后只读生成 schema-v5 三输入报告。
