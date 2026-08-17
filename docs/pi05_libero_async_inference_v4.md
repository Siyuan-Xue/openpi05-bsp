# pi0.5 LIBERO 异步推理与 schema v4

本文说明 `refactor/pi05-libero-bsp-slim` 上的 LIBERO schema-v4 实现、它与权威参考实现的
对应关系，以及推送后应在服务器完成的验证。它不改变 schema v2/v3 的语义，也不把旧产物升级或
混入 v4 报告。

> 验证状态：本次改动只在 macOS 本机做代码与文档开发；未安装运行环境，也未运行测试、导入、
> lint、format、build 或评测。文末命令仅供推送后由服务器接手者执行，均为 **NOT RUN**。

## 权威来源与适用边界

### B-spline Policy（BSP）

- 论文：[BSP paper（arXiv:2607.09648v1）](https://arxiv.org/html/2607.09648v1)。
- 作者仓库固定在 commit
  [`61ed5f42fced971d50a89b46417493790876ccd1`](https://github.com/B-spline-policy/bspline-policy/tree/61ed5f42fced971d50a89b46417493790876ccd1)，
  continuous execution、single-inflight 与 prefetch 的直接参考是该 revision 的
  [`policy_local_bspline.py`](https://github.com/B-spline-policy/bspline-policy/blob/61ed5f42fced971d50a89b46417493790876ccd1/bspline_policy/bspline_policy/scripts/policy_local_bspline.py)；
  FITPACK 分段与 knot repair 的直接参考是
  [`bspline_action.py`](https://github.com/B-spline-policy/bspline-policy/blob/61ed5f42fced971d50a89b46417493790876ccd1/bspline_policy/bspline_policy/common/bspline_action.py)。

本仓库保留 cubic degree 3、`(16, 8)` 参数、前 7 维控制量与第 8 维 knot、前 12 行控制点、
下降 knot 的 `previous + 1e-6` 投影和闭区间内采样。LIBERO 是 delta end-effector action；依论文
Appendix 的 simulation 设置，**不做 segment alignment**，协议固定为
`alignment="disabled_delta_eff"`。因此不能把论文中面向 absolute-pose 的 closest-point/alignment
步骤移植到这里。

BSP async 只借用作者实现中的 single-inflight 与 prefetch 时机。generation/epoch 失效、旧连接退休、
owner 线程 non-daemon、有界 `join` 以及连接/推理/关闭错误通道都是本仓库的 OpenPI/LIBERO transport
实现，不应归因于 BSP 论文或作者代码。

### Real-Time Chunking（RTC）

- 论文：[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/html/2506.07339v2)。
- Physical Intelligence 研究页：[Real-Time Chunking](https://www.pi.website/research/real_time_chunking)。
- 作者 Kinetix 代码固定在 commit
  [`9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/tree/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b)，
  其中 [`src/model.py`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/model.py)
  与 [`src/eval_flow.py`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b/src/eval_flow.py)
  共同作为 model/eval 协议参考。

本实现把完整 guided VJP 接入 OpenPI 的 `Pi0.sample_actions_rtc`，不是在客户端对两个 action chunk
做线程化插值的近似。论文时间变量从 data 到 noise；OpenPI sampler 从 noise 到 data，所以映射后
固定使用 `t = [1.0, 0.8, 0.6, 0.4, 0.2]`，每步
`x_hat = x - t * v`，通过 VJP 得到 guidance，再以 `v_guided = v - gamma * guidance` 和
`x <- x - v_guided / 5` 积分。`gamma=((1-t)^2+t^2)/(t(1-t))` 按 `beta=5` 截断，noise endpoint
也取 5。

LIBERO/OpenPI 的冻结适配是：horizon `H=16`、最早发起位置 `s_min=8`、五步采样 `n=5`、
`beta=5`。delay history 初始化为校准得到的 `d_init`，以后追加每次 guided request 的实际 delay
ticks，只保留最新 10 项，并取其中最大值作为 forecast；它不是“最近 10 次实际延迟”从空队列开始。
服务端保留完整 `(16, 32)` normalized model actions 供下一次引导，但一致性 mask 只覆盖原生动作的
前 `7/32` 维。客户端同时严格校验 `actions:(16,7)` 与 `rtc.model_actions:(16,32)`。

### WebSocket 单 owner

single-owner 合约对照 websockets 13.1 的同步 API 文档验证；该文档明确，同一连接上两个线程并发
调用 `recv()` 会触发 `ConcurrencyError`：
[websockets 13.1 sync client reference](https://websockets.readthedocs.io/en/13.1/reference/sync/client.html)。
这不是把服务器 runtime 钉死在 13.1：`openpi-client` 声明的是 `websockets>=11.0`，当前根 lock
解析到 15.x，而本路径只使用 13.1/15.x 共有的 sync client surface。实际服务器版本必须由下文
preflight 报告，不能从文档假定。
因此连接创建、metadata 接收、request `send/recv` 与 socket close 全部由一个 owner 线程串行拥有；
控制线程只使用 `submit/poll/wait/reset_generation/close`。worker 最多一个 queued/active request，
每次 episode/retry 的 generation reset 会废弃旧 job 并等待旧连接退休，不允许第二个接收者绕开它。

## 四种执行模式

一个 run 只能选择一种冻结模式；同 checkpoint 的正式比较要求四种模式各一个 run。

| `execution_mode` | server family / protocol | 请求与执行 | 校准 |
|---|---|---|---|
| `baseline_sync_n5` | baseline / `baseline_h16_n5_v1` | 每个同步请求都带 schema-only RTC envelope，以固定 n=5 生成 `(16,7)`；执行前 8 步后阻塞重规划 | 无 |
| `baseline_rtc` | baseline / `baseline_rtc_h16_v1` | 首次请求阻塞；达到 `max(8, forecast_delay)` 后后台 guided RTC；结果完成即按实际延迟 cursor 立即替换 | RTC latency |
| `bsp_spline_sync` | BSP / `bsp_spline_h8_v1` | 首次请求阻塞；按 monotonic wall clock 连续采样曲线，闭区间耗尽后阻塞替换 | 无 |
| `bsp_spline_async` | BSP / `bsp_spline_h8_v1` | 首次请求阻塞；曲线剩余时间 `<=` 校准 budget 时单次预取，完成即从新曲线起点替换 | BSP latency |

四种模式都使用确定性 `__openpi_inference_seed` request envelope。baseline 请求还使用
`__openpi_rtc`：bootstrap 只有 `schema_version: 1`；guided 请求还必须有只读 float32
`previous_model_actions:(16,32)`、`8 <= s <= 16`、`0 <= d <= s` 且 `s+d <= 16`。RTC response
必须同时含 native `actions:(16,7)` 和 schema-one `rtc.model_actions:(16,32)`。

BSP response 必须同时保留 legacy `actions:(8,7)` 和严格的 `bsp` sidecar。sidecar 字段精确为
`schema_version, parameters, origin_hz, degree, speedup, alignment`，值分别约束为
`1, (16,8), 10, 3, 1, disabled_delta_eff`。任何缺字段、多字段、shape 不符或 NaN/Inf 都 fail closed。

## 20 Hz 控制与异步时序

控制周期固定为 20 Hz，即 `50_000_000 ns`；视频固定为 40 fps。dataset 10 fps 和 source demo
20 Hz 只作为 provenance，不能拿来驱动 evaluator。dummy phase 也按 20 Hz pacing，但不计入 episode
timeline。

控制器采用 no-catch-up：每个真实 `env.step` 开始后，把下一 deadline 重锚到
`actual_start + 50 ms`。如果推理完成得晚，记录 inference latency 与 control stall，只等待当前缺失
动作，不补跑已经错过的 step，也不连发动作追赶旧绝对时钟。选中视频通常每个 control frame 保持
两帧；只有明确记录并选择显示的 inference wait 才插入冻结帧，视频构造不会增加 simulator step、
dummy action 或 sleep。

## 校准：不是固定 60 ms

只有两个异步模式在写 manifest 之前校准；同步模式的 `latency_calibration` 必须为 null。校准绑定
checkpoint identity、完整 server metadata fingerprint、首个 suite/task/init-state 的 canonical
observation fingerprint 和确定性 seed namespace。RTC 先做一次不计入统计的 bootstrap，以得到可链式
guided 的 model actions；然后两种异步模式都执行：

1. 5 次 warmup（保留原始 latency 与 request fingerprint，但不参与分位数）；
2. 20 次 measurement；
3. 对 20 个 `monotonic_ns` latency 使用 nearest-rank p95。

20 个样本的 nearest-rank 是升序后的第 `ceil(0.95 * 20)=19` 个值（1-based），不是插值分位数。
`p95` 没有 60 ms 常量：

- RTC：`d_init = ceil(p95 / 50 ms)`；若 `d_init > 8`，校准直接失败，不能静默截断为 8。
- BSP async：`budget = ceil(p95 / 50 ms) * 50 ms`；每个候选 curve 安装前都要求
  `budget < usable_curve_duration`，不满足则失败，不能提前污染 active plan。

校准过程中基础设施错误最多允许两次完整重试；重试必须 reset generation 并重新验证 exact server
identity。malformed response、协议/metadata/checkpoint 漂移属于当前 policy/identity failure，不可当成
网络抖动重试。

## capability 与 fail-closed 边界

连接后的 metadata 必须包含精确 schema-one `__openpi_inference_capabilities`：

```text
schema_version, action_representation, model_action_horizon,
model_action_dim, supported_protocols
```

baseline 要求 `native, 16, 32` 且协议列表精确为
`[baseline_h16_n5_v1, baseline_rtc_h16_v1]`；BSP 要求 `bsp, 16, 32` 且列表精确为
`[bsp_spline_h8_v1]`。metadata 全量 canonical fingerprint 写入 manifest，并在校准、每个 request
completion 和重连后再次比对。能力缺失、family 错误、连接 identity 变化、过期/取消结果、负 latency、
时钟倒退或响应 shape 错误都不得退回旧 chunk 或零动作掩盖。

基础设施异常按同一 episode seed 最多重试两次；policy failure 计入失败 episode。episode 完成时仍在途
的结果只可通过 generation reset 废弃，不能记为 latency/activation。worker/environment close 失败是
run-fatal，即使已有 primary error 也必须同时保留 cleanup failure。

## schema-v4 产物与报告隔离

每个 v4 run 独占一个新且为空的 output directory。`ArtifactWriterV4` 以原子替换写入其中的
JSON/JSONL 结构化产物；MP4 路径由 `VideoSelectorV4` 预留，随后由编码器直接写入并通过
`video_audit.jsonl` 复核：

- `manifest.json`：冻结 mode、capability/metadata、checkpoint/runtime/cache identity 与校准原始样本；
- `episodes.jsonl`：episode/retry 状态和 request、latency、activation、underflow、stall 时序；
- `summary.json`：由 episode 与 artifact errors 派生的完整性汇总；
- `video_audit.jsonl` 与 `videos/`：只记录被选择视频的计划帧、可选 stall 帧、最多一帧的零步 artifact
  padding 与编码后复核；
- `artifact_errors.jsonl`：仓库内 v4 writer 初始化时总会创建（可以为空）；loader 为兼容外部/既有 v4
  目录允许该文件缺失，但只要存在且非空，正式报告就拒绝该 run。

v4 loader 严格拒绝 duplicate JSON keys、NaN/Inf、字段集合或汇总不一致、infrastructure incomplete、
视频审计语义不一致和任何 artifact error。formal loader 计算并保留四个必需文件
`manifest.json`、`episodes.jsonl`、`summary.json`、`video_audit.jsonl` 的 SHA256；它不对 MP4/video
文件计算或验证 hash，video 只通过持久化的 audit 字段做语义校验。它不调用 v2/v3 producer/reader；旧
`main.py`、旧 manifest 与旧 reporter 保持只读语义，不能自动升级。

单 checkpoint 报告需要精确四个 run；正式五 checkpoint 报告需要
`0/1000/2000/5000/10000 x 4 modes = 20 runs`。primary paired deltas 只有
`baseline_rtc - baseline_sync_n5` 和 `bsp_spline_async - bsp_spline_sync`，episode identity 必须逐项
匹配；输出为 `comparison_v4.json`、`learning_curve_v4.csv` 和 `report_v4.md`。

## 关键文件

| 文件 | 责任 |
|---|---|
| `src/openpi/models/pi0.py` | OpenPI 时间方向上的 RTC 五步 guided VJP sampler |
| `src/openpi/policies/policy.py` | inference envelope、capability metadata、RTC normalized sidecar |
| `src/openpi/training/bsp.py` | 权威 commit 对应的 BSP target/cache 移植 |
| `packages/openpi-client/src/openpi_client/async_inference.py` | single-owner、single-inflight、generation reset、non-daemon/join/error channel |
| `packages/openpi-client/src/openpi_client/rtc.py` | RTC chunk/cursor/delay-history 客户端状态机 |
| `packages/openpi-client/src/openpi_client/bsp_spline.py` | 无 SciPy runtime 依赖的 NumPy continuous curve 与 wall-clock plan |
| `packages/openpi-client/src/openpi_client/libero_control_v4.py` | 四 mode、校准、capability 与 scheduler fail-closed 合约 |
| `packages/openpi-client/src/openpi_client/libero_video_timing_v4.py` | 20 Hz/40 fps request、stall、video timeline 校验 |
| `packages/openpi-client/src/openpi_client/libero_eval_v4.py` | schema-v4 records、retry 与原子 artifact writer |
| `packages/openpi-client/src/openpi_client/libero_report_v4.py` | 独立 v4 loader、四 mode/五 checkpoint 配对报告 |
| `examples/libero/main_v4.py` | `ArgsV4` 入口与完整 evaluator orchestration |

## 推送后的服务器验证交接（全部 NOT RUN）

以下命令不由本机或本次改动执行。接手者应先在服务器进入已推送 commit 的**干净 checkout**，再复用
既有 runbook 中已安装的两个环境；不要在 macOS 本机安装依赖。

canonical runbook 最后用 `pip install --no-deps -e packages/openpi-client`，它只安装 editable package，
**不保证** client 环境已经有 `pytest`、`msgpack` 或 `websockets`。本任务未获授权在本机安装任何
依赖；服务器接手者也应先执行下面的版本/import gate，缺包或版本不符就停止，再按服务器变更流程
显式补齐环境，不能跳过 gate 后把 import failure 误判为代码回归。

### OpenPI 根环境：Python 3.11（NOT RUN）

版本/import preflight（NOT RUN）：

```bash
cd "$BSP_REPO_DIR"
"$OPENPI_PY" - <<'PY'
import sys
from importlib import metadata

assert sys.version_info[:2] == (3, 11), sys.version
import msgpack
import pytest
import websockets
for distribution in ("pytest", "msgpack", "websockets"):
    print(distribution, metadata.version(distribution))
from openpi.models import pi0
from openpi.policies import policy
from openpi.training import bsp
print("openpi_root_import_gate=PASS")
PY
```

测试合同（NOT RUN）：

```bash
cd "$BSP_REPO_DIR"
"$OPENPI_PY" -m pytest -q \
  src/openpi/models/pi0_test.py \
  src/openpi/policies/policy_seed_test.py \
  src/openpi/policies/libero_policy_test.py \
  scripts/serve_policy_test.py \
  src/openpi/training/bsp_test.py \
  src/openpi/training/bsp_dataset_test.py \
  src/openpi/training/loader_resume_test.py \
  src/openpi/training/train_planning_test.py \
  scripts/prepare_libero_bsp_test.py
```

按待测 family 启动一个 policy server；baseline 两种 mode 使用 baseline checkpoint，BSP 两种 mode
使用 BSP checkpoint。下面只是 handoff 模板（NOT RUN）：

```bash
cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config "$CONFIG_NAME" \
  --policy.dir "$CHECKPOINT"
```

### LIBERO client 环境：Python 3.8（NOT RUN）

版本/import preflight（NOT RUN）；这里打印实际 websockets 版本，而不是假定为 13.1：

```bash
cd "$BSP_REPO_DIR"
env PYTHONPATH="$LIBERO_PYTHONPATH" "$LIBERO_PY" - <<'PY'
import sys
from importlib import metadata

assert sys.version_info[:2] == (3, 8), sys.version
import msgpack
import pytest
import websockets
for distribution in ("pytest", "msgpack", "websockets"):
    print(distribution, metadata.version(distribution))
from openpi_client import async_inference
from openpi_client import libero_control_v4
from openpi_client import libero_eval_v4
from openpi_client import libero_report_v4
from openpi_client import libero_video_timing_v4
from openpi_client import websocket_client_policy
from examples.libero import main_v4
print("libero_client_import_gate=PASS")
PY
```

新旧协议共同测试合同（NOT RUN）；v3 tests 在此作为“不被 v4 改写”的回归门禁：

```bash
cd "$BSP_REPO_DIR"
env PYTHONPATH="$LIBERO_PYTHONPATH" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" -m pytest -q \
    packages/openpi-client/src/openpi_client/inference_test.py \
    packages/openpi-client/src/openpi_client/msgpack_numpy_test.py \
    packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
    packages/openpi-client/src/openpi_client/async_inference_test.py \
    packages/openpi-client/src/openpi_client/rtc_test.py \
    packages/openpi-client/src/openpi_client/bsp_spline_test.py \
    packages/openpi-client/src/openpi_client/libero_video_timing_test.py \
    packages/openpi-client/src/openpi_client/libero_eval_test.py \
    packages/openpi-client/src/openpi_client/libero_eval_video_test.py \
    packages/openpi-client/src/openpi_client/libero_report_test.py \
    packages/openpi-client/src/openpi_client/libero_control_v4_test.py \
    packages/openpi-client/src/openpi_client/libero_video_timing_v4_test.py \
    packages/openpi-client/src/openpi_client/libero_eval_v4_test.py \
    packages/openpi-client/src/openpi_client/libero_eval_v4_video_test.py \
    packages/openpi-client/src/openpi_client/libero_report_v4_test.py \
    scripts/libero_eval_test.py \
    scripts/libero_eval_v4_test.py
```

评测入口必须是 `examples/libero/main_v4.py` 的 `ArgsV4`，每次只写一个唯一、空的 output directory。
正式 run 还必须提供真实的 checkpoint、norm、container 和 BSP cache identity；以下为 handoff 模板，
不是已执行命令：

```bash
cd "$BSP_REPO_DIR"
env PYTHONPATH="$LIBERO_PYTHONPATH" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" examples/libero/main_v4.py \
    --args.host "$POLICY_HOST" \
    --args.port 8000 \
    --args.execution-mode "$EXECUTION_MODE" \
    --args.task-suite-name all \
    --args.control-freq 20 \
    --args.video-fps 40 \
    --args.output-dir "$OUTPUT_DIR" \
    --args.config-name "$CONFIG_NAME" \
    --args.checkpoint-step "$STEP" \
    --args.checkpoint "$CHECKPOINT" \
    --args.norm-hash "$NORM_HASH" \
    --args.container-digest "$CONTAINER_DIGEST"
```

当 `$EXECUTION_MODE` 为 `bsp_spline_sync` 或 `bsp_spline_async` 时，还要追加
`--args.bsp-cache-hash "$BSP_CACHE_HASH"` 与
`--args.bsp-cache-manifest-fingerprint "$BSP_FINGERPRINT"`；baseline mode 必须省略二者。先用一个
checkpoint 的四个 mode 完成服务器 smoke/正式验证，再决定是否生成五 checkpoint 的 20-run 报告。

`libero_report_v4.py` 没有 CLI；交接时必须调用现有 Python API
`write_five_checkpoint_report_v4(run_dirs, *, output_dir=Path(...))`。同一 API 接受精确 4 个 run
（单 checkpoint 四 mode）或精确 20 个 run（五个固定 checkpoint x 四 mode）。下面的 bash array
必须由接手者填入 4 或 20 个唯一目录，report output 必须不存在或为空（NOT RUN）：

```bash
RUN_DIRS=(
  "/path/to/run-1"
  "/path/to/run-2"
  "/path/to/run-3"
  "/path/to/run-4"
)
export REPORT_OUTPUT="/path/to/new-or-empty-v4-report"

cd "$BSP_REPO_DIR"
env PYTHONPATH="$LIBERO_PYTHONPATH" \
  "$LIBERO_PY" -c 'from pathlib import Path; import sys; from openpi_client.libero_report_v4 import write_five_checkpoint_report_v4; write_five_checkpoint_report_v4([Path(value) for value in sys.argv[2:]], output_dir=Path(sys.argv[1]))' \
  "$REPORT_OUTPUT" "${RUN_DIRS[@]}"
```

这里不声明任何 preflight、测试、评测或报告命令已经成功。
