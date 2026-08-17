# LIBERO-only WebSocket inference

本仓库只支持 LIBERO 的双环境远程推理。OpenPI/JAX 和 LIBERO 的依赖版本不兼容，因此不要
合并环境：

```text
Python 3.11.9                         Python 3.8.20
OpenPI + JAX + pi0.5 checkpoint      MuJoCo + robosuite + LIBERO
scripts/serve_policy.py       <----> openpi-client + examples/libero/main.py
                         WebSocket
```

完整路径、EGL 和评测协议见 [服务器 runbook](pi05_libero_bsp_phase1_server.md)；本页只说明
通信边界。

schema-v4 的四模式异步推理、权威算法映射、动态 latency calibration 与服务器验证交接见
[pi0.5 LIBERO 异步推理与 schema v4](pi05_libero_async_inference_v4.md)。v4 使用独立入口与产物，
不会改变本页描述的旧 evaluator/schema-v3 语义。

## `openpi-client` 是什么

`packages/openpi-client` 是轻量 Python 包，不含模型权重或 JAX。它保留：

- `websocket_client_policy.py`：连接 policy server，获取 metadata，发送 inference request；
- `msgpack_numpy.py`：无损编码 NumPy arrays；
- `image_tools.py`：将相机输入 resize/uint8；
- `inference.py`：可选确定性 flow-noise seed 字段；
- `libero_eval.py`：episode 身份、错误分类、两次基础设施重试和审计产物；
- `libero_report.py`：baseline/BSP 五个里程碑的配对报告。

已删除的通用 agent runtime 和 action chunk broker 不参与 LIBERO evaluator。

## 启动 policy server

在 OpenPI Python 3.11 环境中，从仓库根目录运行。官方校准：

```bash
export BSP_WORK=/root/openpi-bsp-work
export BSP_REPO_DIR="$BSP_WORK/repo/openpi05-bsp"
export OPENPI_PY="$BSP_WORK/venvs/openpi/bin/python"
export OPENPI_DATA_HOME="$BSP_WORK/cache/openpi"
export JAX_COMPILATION_CACHE_DIR="$BSP_WORK/cache/jax"

cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
```

训练 checkpoint 只替换 config 和目录：

```bash
"$OPENPI_PY" scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero_baseline_lora_h16 \
  --policy.dir "$BSP_WORK/experiments/checkpoints/pi05_libero_baseline_lora_h16/phase1-short10k-seed42-baseline/1000"
```

不要使用与 checkpoint 不匹配的 config；policy construction 会从 checkpoint assets 读取对应
norm stats。一次 GPU 只运行一个 policy/training 作业。

## LIBERO request schema

Python 3.8 client 每次 replan 发送：

```python
observation = {
    "observation/image": agentview_rgb_uint8,
    "observation/wrist_image": wrist_rgb_uint8,
    "observation/state": state_float_array,
    "prompt": task_description,
    "__openpi_inference_seed": deterministic_uint32,  # request envelope; not a model observation
}
result = client.infer(observation)
actions = result["actions"]
```

`examples/libero/main.py` 会把 LIBERO 相机数组旋转 180°以匹配训练，再 resize 到 224×224。
state 不在 client 归一化；server 使用 checkpoint norm stats。确定性 seed 由
`(suite, task, init_state, replan_index)` 和 eval seed 42 稳定派生，使 A/B 在同一配对 episode
使用相同 flow noise。未提供 seed 时仍兼容 policy 的 stateful RNG。

## 输出协议

| variant | 模型原始目标 | server 返回 | evaluator 执行 |
|---|---|---|---|
| 官方校准 | baseline horizon 10 | `(10, 7)` | 前 8 步 |
| phase-one baseline | baseline horizon 16 | `(16, 7)` | 前 8 步 |
| phase-one BSP | 16×8 spline parameters | 解码后的 `(8, 7)` | 全部 8 步 |

BSP output transform 在 quantile 反归一化后：

1. 取前 7 个 control 通道和第 8 个 knot 通道；
2. 将下降 knot 投影为 `previous + 1e-6`，保留重复边界 knot；
3. 使用 16 knots 和前 12 个 control points；
4. 只在 `[knots[3], knots[-4]]` 内等距解码 8 个动作，禁止外推；
5. shape 错误、NaN 或无有效区间分类为 policy failure。

两组在原生 10 Hz 下执行 8 步后重新请求。第一阶段不做 2×/4× 时间缩放、异步 action
broker、segment alignment 或 gripper threshold。

## 在 simulator 环境连接

```bash
export BSP_WORK=/root/openpi-bsp-work
export BSP_REPO_DIR="$BSP_WORK/repo/openpi05-bsp"
export LIBERO_PY="$BSP_WORK/venvs/libero-py38/bin/python"
export LIBERO_PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero:$BSP_REPO_DIR/third_party/libero/libero"
export EGL_VENDOR_JSON="$BSP_WORK/cache/egl/10_nvidia.json"

cd "$BSP_REPO_DIR"
env \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" examples/libero/main.py --help
```

正式运行必须补齐 evaluator 的 config/checkpoint/step/code SHA/dataset revision/norm hash/runtime
digest，以及 BSP 时的 sidecar hash/fingerprint。不要用空身份生成验收结果；完整命令见
[LIBERO evaluator README](../examples/libero/README.md)。

## 错误边界

- 连接、超时、simulator reset/step 和网络错误是 infrastructure failure；相同 seed 最多重试两次。
- 非法 action shape、NaN/Inf、BSP 解码失败是 policy failure，计入成功率的失败 episode。
- 两次基础设施重试仍失败、视频 artifact 写入失败或 manifest 不完整时，整个 run
  `acceptance_complete=false`。
- client 不应把异常吞掉并返回零动作；server 也不应为 shape 不匹配自动截断。

如果 policy server 与 simulator 不在同一主机，只开放明确的受控网络路径。当前服务没有把
WebSocket 当作公网鉴权边界，不应直接暴露到互联网。
