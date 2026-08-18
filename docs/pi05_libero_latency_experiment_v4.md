# π0.5 + LIBERO schema-v4 真实延迟实验

本文是服务器 `main` 上四模式延迟实验的执行合同。它只研究推理结果更晚可用时，
同步和异步调度对控制停顿与吞吐的影响；不改变模型权重、动作值、20 Hz 动力学、
episode 步数上限或成功判定。

保护基线为 `d3702fccc85fbbe2ff4df691ce9c5c7a11964b55`，annotated tag 为
`pre-async-latency-main-d3702fc`。异步原始提交链保留在
`codex/refactor-async-integration`；main 上的语义移植提交在 message 中记录来源 SHA。
禁止把瘦身分支整体合并到 main，也禁止 rebase 或 force-push main。

## 四模式与时钟

| 模式 | checkpoint family | 执行协议 |
|---|---|---|
| `baseline_sync_n5` | baseline LoRA H16 | 固定五步采样，同步重规划 |
| `baseline_rtc` | baseline LoRA H16 | RTC guided 预取 |
| `bsp_spline_sync` | BSP LoRA H8 | 连续样条，同步替换 |
| `bsp_spline_async` | BSP LoRA H8 | 连续样条预取和即时替换 |

四个独立时钟保持不变：dataset 为 10 FPS，专家来源环境为 20 Hz，评测控制为
20 Hz，MP4 为 40 FPS。只有评测控制频率进入动力学；视频频率只影响编码。

## 延迟注入

入口参数为：

```text
--args.synthetic-latency-target-ms 0|100|200|300|850
```

`0` 表示原生 H20 延迟。非零值表示从请求提交到结果对 scheduler 可见至少达到
目标时间。WebSocket response 先正常返回，single-owner worker 再等待到目标 deadline，
最后才发布 outcome。原始请求已经比目标慢时不截短。

每个成功请求分别记录：

- `raw_inference_latency_ns`：请求提交到 WebSocket response 返回；
- `synthetic_delay_ns`：response 返回后实际补足的时间，包含 sleep overshoot；
- `effective_inference_latency_ns`：请求提交到结果对 scheduler 可见；
- activation、action underflow 和真实 `control_stall`。

必须满足：

```text
raw + synthetic = effective = completed_monotonic_ns - submitted_monotonic_ns
```

异步校准使用 effective latency。一次不计时的 RTC bootstrap 之后，5 次 warmup
不进入分位数，20 次 measurement 用 nearest-rank p95 形成调度预算。注入不修改
request observation、seed、response 或 action 数组。

本方法模拟“推理结果更晚可用”，适合测量 latency hiding 和吞吐；它不模拟低算力
GPU 的功耗、显存、算子数值差异或多请求并发上限。

## 视频

正式录像使用：

```text
--args.control-freq 20
--args.video-fps 40
--args.video-show-inference-waits
```

不做十倍视频插帧。视频只为真实 `control_stall` 增加对应的 40 FPS 冻结帧；异步
request 有 latency 但没有欠载时不冻结。每帧持续显示一行：

```text
Cumulative inference wait: X.XX s
```

首帧显示 `0.00 s`。stall 帧按 40 FPS 递增，正常运动帧保留累计值，下次 stall
继续累计。文字为白色并使用一像素深色描边，不绘制黑色矩形。编码器逐帧写入，
不会为长 episode 先构造完整扩展帧数组。录像开关只位于 artifact 阶段，不参与
rollout，因此不能改变 action、steps、done、status 或 success。

## 容量扫描

延迟点为 `100/200/300/850 ms`。每个延迟点运行四模式；每个 run 固定四个 suite
的 task 0，每个 task 5 个 trial，共 20 episodes：

```text
--args.task-suite-name all
--args.task-ids 0
--args.num-trials-per-task 5
```

输出目录必须同时包含 latency 和 mode，例如：

```text
capacity/latency-300ms/baseline_rtc
```

一次只允许一个 policy server 和一个 evaluator。850 ms 超出 baseline RTC 的八个
20 Hz tick 容量时，校准必须拒绝，不能把 delay 截到 8、延长 horizon 或降低控制
频率。BSP async 仍需按返回曲线的 usable duration 做 fail-closed 检查；若拒绝，
保留日志并把它记录为容量边界，不创建伪造的成功 run。

20-episode 结果是扫描结果，不得传给只接受完整 2000-episode grid 的 formal loader。

## 正式实验

只运行两个延迟条件：

1. 原生延迟：`--args.synthetic-latency-target-ms 0`；
2. 300 ms：`--args.synthetic-latency-target-ms 300`。

每个条件运行四模式，每个 run 使用：

```text
--args.task-suite-name all
--args.num-trials-per-task 50
```

总计 8 个正式 run。原生条件和 300 ms 条件必须分别调用四模式 reporter，不能在
同一个比较中混合 synthetic target。正式 loader 要求每个 run 精确 4 suites、
40 tasks、2000 denominator-eligible episodes、零 infrastructure incomplete、零
artifact error，并核对全部结构化产物。

## evaluator 命令模板

下列命令在一个全新且为空的 `$OUTPUT_DIR` 上执行。`main_v4.py` 自动解析 clean
checkout HEAD，禁止传手写 code SHA：

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$SCHEMA4_REPO/src:$SCHEMA4_REPO:$SCHEMA4_REPO/packages/openpi-client/src:$SCHEMA4_REPO/third_party/libero" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" "$SCHEMA4_REPO/examples/libero/main_v4.py" \
    --args.host 127.0.0.1 \
    --args.port 8000 \
    --args.execution-mode "$EXECUTION_MODE" \
    --args.synthetic-latency-target-ms "$LATENCY_MS" \
    --args.task-suite-name all \
    --args.num-trials-per-task "$TRIALS_PER_TASK" \
    --args.control-freq 20 \
    --args.video-fps 40 \
    --args.video-show-inference-waits \
    --args.output-dir "$OUTPUT_DIR" \
    --args.config-name "$CONFIG_NAME" \
    --args.checkpoint-step 10000 \
    --args.dataset-revision v2.0 \
    --args.norm-hash "$NORM_HASH" \
    --args.checkpoint "$CHECKPOINT" \
    --args.container-digest "$HOST_DIGEST" \
    --args.train-seed 42 \
    --args.eval-seed 42
```

容量扫描额外传 `--args.task-ids 0`。BSP 两个模式还必须传：

```text
--args.bsp-cache-hash "$BSP_CACHE_HASH"
--args.bsp-cache-manifest-fingerprint "$BSP_FINGERPRINT"
```

baseline 两模式禁止传 BSP identity。

## 服务器门禁顺序

1. 新建独立 clean worktree，核对精确 main SHA；不覆盖 d3702fc 目录。
2. 核对 Python 3.11 OpenPI、Python 3.8 LIBERO、LIBERO 子模块和 policy capability。
3. 运行 CPU/client/model 目标测试。
4. EGL reset/render/step。
5. 四模式原生延迟单 episode smoke。
6. 四模式 300 ms 单 episode smoke。
7. 同一同步输入分别关闭/开启录像，核对 action、steps、done、status、success 一致。
8. 回读 MP4，核对 40 FPS、累计文字、无黑框、帧数、时长和一帧误差。
9. 运行 16 个容量扫描条件。
10. 只在前述门禁通过后运行 8 个正式 run。

BSP diagnostics 必须由新 clean main SHA 对现有 sidecar 执行到一个新路径。只做 verify，
不重新拟合、不覆盖旧 diagnostics、不修改 checkpoint/norm/sidecar。

## 正式指标与结论边界

四模式报告除成功率外，还输出：

- raw、synthetic 和 effective inference latency；
- control stall 总量、均值；
- underflow 次数与持续时间；
- `max(0, effective latency - stall) / effective latency` latency hiding ratio；
- control steps；
- episode monotonic wall time 与由其推导的 episodes/min。

主要结论是“延迟隐藏与吞吐收益”。成功率、控制步数和 suite/task 结果用于确认策略
质量没有异常退化。由于 underflow 时本 evaluator 暂停仿真，不能把结论直接表述成
真机抽搐或物理成功率改善。

若服务器验证失败，停止实验并保留日志和产物。修复通过新的 main commit 或 revert
commit 完成；不得 rebase、force-push 或改写已发布历史。
