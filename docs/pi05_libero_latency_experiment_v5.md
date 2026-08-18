# π0.5 + LIBERO schema-v5 随机延迟三模式实验

本文冻结 schema-v5 的实现、服务器门禁和后续正式实验边界。实现基线是
`main@3861651ca0f63895926c491dd21dda2194e643a7`，保护 tag 是
`pre-random-latency-schema4-3861651`。实现提交只向前追加；禁止 rebase、
force-push 或改写已发布历史。

本轮复用现有 10K checkpoint、norm stats 和 BSP sidecar，不重新训练、不重新拟合。
候选代码通过服务器门禁后先停止；没有用户再次明确批准，不启动正式 2000-episode
评测，也不恢复监控任务。

## 三种正式模式

| 模式 | 模型 | 调度语义 |
|---|---|---|
| `baseline_async` | baseline LoRA H16 | 普通异步 raw chunk；跳过已过期前缀后直接切换，允许 segment seam |
| `baseline_rtc` | baseline LoRA H16 | 与普通 async 同一步发请求，携带连续性指导并用 RTC 拼接 |
| `bsp_spline_async` | BSP LoRA H8 | 连续样条异步预取和激活 |

三者共享同一延迟采样、注入位置、20 Hz 控制、40 FPS 视频、seed 42 和 step 10000。
schema-v5 不提供同步模式、固定延迟扫描或 target=0 正式条件。

## 确定性配对随机延迟

正式分布固定为：

```text
distribution = normal
mean_ns = 300000000
stddev_ns = 60000000
seed = 42
sampler_version = sha256_box_muller_v1
negative_policy = deterministic_resample
```

样本键为：

```text
namespace / seed / suite / task_id / trial_index / request_ordinal
```

正式 namespace 为 `formal`。校准使用独立 `calibration/...` namespace，不消耗正式
样本。采样器只使用 Python 标准库、SHA-256 和 Box–Muller；负值通过增加 attempt 后
重新哈希。结果冻结为整数纳秒，因此三个模式对相同 episode/request ordinal 得到完全
相同的目标延迟，不依赖平台随机数实现。

worker 在 WebSocket response 返回后、把 outcome 发布给 scheduler 前补足延迟：

```text
scheduled_effective_latency = max(raw_latency, sampled_target)
requested_synthetic_delay = scheduled_effective_latency - raw_latency
observed_effective_latency = raw_latency + observed_synthetic_delay
latency_overshoot = observed_effective_latency - scheduled_effective_latency
```

真实线程可能略晚于目标 deadline 被操作系统唤醒，因此只要求
`observed_effective_latency >= scheduled_effective_latency`，不要求二者纳秒级相等。每个
request 都记录 sample key、sampled target、raw、requested synthetic、observed synthetic、
observed effective 和 overshoot。Calibration 与报告使用 observed effective；固定 400 ms
调度预算仍来自理论分布，不被偶发唤醒超调改变。
该方法模拟“策略结果更晚可用”，不声称模拟低算力 GPU 的显存、功耗或并发能力。

## Baseline 公平调度

`baseline_async` 和 `baseline_rtc` 调用同一个启动门禁：

```text
MIN_START = 8
forecast_delay = 8 control ticks
cursor >= max(MIN_START, forecast_delay)
cursor + forecast_delay <= 16
```

Normal(300 ms, 60 ms) 的理论 p95 为 398.691217617 ms，固定向上对齐为 400 ms，
在 20 Hz 下即 8 ticks。5 次 warmup 与 20 次 measurement 只产生 empirical 审计和
告警，不能改变这个预算，也不能随机阻断实验。

普通 async 不发送 `previous_model_actions`、`s` 或 `d`；response 激活时跳过请求以来
已经执行的 raw chunk 前缀。RTC 发送连续性指导并保持 RTC 拼接。动作耗尽而 outcome
仍不可用时，才记录 underflow 和真实 control stall。

每次非初始计划激活记录六维 arm action 的 L2 jump、max-absolute jump 以及 gripper
absolute jump。初始计划没有上一段，因此不生成 seam 事件。

## 时钟与视频

四个时钟保持独立：dataset 10 FPS、来源专家环境 20 Hz、评测控制 20 Hz、视频
40 FPS。dataset/source 频率不参与视频帧数计算，也不能为了视觉效果修改控制频率。

视频只显示真实 control stall，不放大 inference latency。异步请求有 latency 但没有
欠载时，不插入冻结帧。每帧持续显示：

```text
Cumulative inference wait: X.XX s
```

首帧为 `0.00 s`；stall 期间按 40 FPS 累加，正常运动时保持，下一次 stall 继续累加。
文字使用白色和细暗色描边，不绘制黑框。编码保持流式；录像开关不得改变 action、
steps、done、status 或 success。

## evaluator 命令身份

候选 worktree 中使用 `examples/libero/main_v5.py`。CLI 不暴露固定 target：

```text
--args.execution-mode baseline_async|baseline_rtc|bsp_spline_async
--args.task-suite-name all
--args.num-trials-per-task 50
--args.control-freq 20
--args.video-fps 40
--args.video-show-inference-waits
```

正式 evaluator 自动解析 clean checkout HEAD，禁止手写 code SHA。所有服务器进程必须
显式设置候选 worktree 的 `PYTHONPATH`、`PYTHONDONTWRITEBYTECODE=1` 和
`PYTHONNOUSERSITE=1`，避免旧 editable install 或 `.pyc` 污染身份。

BSP 模式还必须传现有 sidecar 的 SHA-256 和 manifest fingerprint；baseline 模式必须
保持这两个字段为 null。新候选 SHA 只对既有 sidecar做非破坏性 verify，并写入新的
schema-v5 diagnostics 路径。

## 服务器门禁（正式实验前停止）

1. 从 `validate/schema5-random-latency` 在独立 clean worktree 检出精确候选 SHA。
2. 核对 Python 3.11 OpenPI、Python 3.8 LIBERO、固定 LIBERO gitlink 和 policy capability。
3. 运行完整 CPU/client 目标测试、Ruff、compileall 和 Python 3.8 语法检查。
4. 运行 EGL reset/render/step。
5. 三模式各运行一个 episode smoke，并核对配对 sample 序列。
6. 核对普通 async 与 RTC 在相同 control step 发起后台请求，但 request envelope 不同。
7. 回读 MP4：40 FPS、单行累计文字、无黑框、真实一倍 stall、帧数和一帧容差。
8. 对同一输入核对录像开关前后的 action/status/success 完全一致。
9. 精确核对 policy PID/starttime/cmdline，普通 SIGTERM；确认 evaluator、GPU 和 8000
   端口全部释放。

任一门禁失败即停止，保留日志和产物，不重试改参、不删除证据、不更新 `origin/main`。
全部通过后，只把同一个候选提交 fast-forward 到 `origin/main`，随后停止。

## 用户再次批准后的正式实验

只有再次明确批准，才串行运行三组：

```text
baseline_async
baseline_rtc
bsp_spline_async
```

每组固定 4 suites × 10 tasks × 50 trials = 2000 episodes。正式 reporter 恰好接受
三个 schema-v5 输入，输出：

```text
comparison_v5.json
task_metrics_v5.csv
report_v5.md
```

报告固定比较：

- `baseline_rtc - baseline_async`；
- `bsp_spline_async - baseline_async`；
- `bsp_spline_async - baseline_rtc`。

指标包括 suite/task/macro success、seam jump、underflow、effective latency、真实 stall、
latency hiding ratio、control steps、episode wall time、吞吐和视频 timing gate。结论只限于
随机延迟下的延迟隐藏、轨迹衔接和吞吐；由于 underflow 时仿真暂停，不能直接解释为
真机动力学改善。
