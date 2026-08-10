# LIBERO evaluator

本目录是专项仓库唯一 simulator 示例。它用锁定的
[`third_party/libero`](../../third_party/libero) 在 Python 3.8.20 中运行 MuJoCo/robosuite，
通过 `openpi-client` 请求 Python 3.11.9/JAX policy server。完整环境、数据和安全门禁见
[服务器 runbook](../../docs/pi05_libero_bsp_phase1_server.md)。

## 文件

| 文件 | 用途 |
|---|---|
| `main.py` | 四套件 evaluator、确定性 seed、错误分类、逐 episode artifact |
| `requirements.in` | simulator 直接依赖来源 |
| `requirements.txt` | Python 3.8 锁定安装输入 |

只初始化 LIBERO gitlink：

```bash
git submodule update --init third_party/libero
test "$(git -C third_party/libero rev-parse HEAD)" = \
  f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

更新 `requirements.txt` 时，PyTorch wheel source 仍需显式指定
`--extra-index-url https://download.pytorch.org/whl/cu113`；不要在 OpenPI Python 3.11 环境中
安装这套 simulator requirements。

仓库级便携环境的创建入口分别是：

```bash
uv sync --python 3.11
uv venv --python 3.8 examples/libero/.venv
```

阿里云 DSW 的已验证布局不使用仓库内 `.venv`，而是按 canonical runbook 把两个环境固定到
`/root/openpi-bsp-work/venvs/openpi` 与 `/root/openpi-bsp-work/venvs/libero-py38`。上面两行只说明
Python 版本边界；服务器安装、锁文件和绝对路径以 runbook 为准。

## 两个终端

共同变量：

```bash
export BSP_WORK=/root/openpi-bsp-work
export BSP_REPO_DIR="$BSP_WORK/repo/openpi05-bsp"
export OPENPI_PY="$BSP_WORK/venvs/openpi/bin/python"
export LIBERO_PY="$BSP_WORK/venvs/libero-py38/bin/python"
export OPENPI_DATA_HOME="$BSP_WORK/cache/openpi"
export EGL_VENDOR_JSON="$BSP_WORK/cache/egl/10_nvidia.json"
export EXPERIMENTS_DIR="$BSP_WORK/experiments"
export EVAL_BASE="${EXPERIMENTS_DIR}/eval"
export CODE_SHA="$(git -C "$BSP_REPO_DIR" rev-parse HEAD)"
export LIBERO_DATASET_REVISION=v2.0
export LIBERO_PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero:$BSP_REPO_DIR/third_party/libero/libero"
export HOST_RUNTIME_DIGEST="sha256:$(
  {
    "$OPENPI_PY" --version
    "$LIBERO_PY" --version
    git -C "$BSP_REPO_DIR/third_party/libero" rev-parse HEAD
    sha256sum "$EGL_VENDOR_JSON"
  } | sha256sum | awk '{print $1}'
)"
```

终端 A 启动 checkpoint：

```bash
cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
```

终端 B 使用固定 EGL 环境调用 evaluator：

```bash
cd "$BSP_REPO_DIR"
mkdir -p "$BSP_WORK/cache/libero-config"
```

实际 client 命令使用下面展示的 `env KEY=value` 形式，避免把 simulator 变量污染到 OpenPI
服务端 shell。

## task 0 × 1 EGL / WebSocket 冒烟

先从官方 checkpoint 内实际文件派生 norm hash，不使用占位符：

```bash
export OFFICIAL_CHECKPOINT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
export OFFICIAL_NORM="$OFFICIAL_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
export OFFICIAL_NORM_HASH="$(sha256sum "$OFFICIAL_NORM" | awk '{print $1}')"
export SMOKE_OUTPUT="$EVAL_BASE/official-h10-libero-spatial-task0-one"
test ! -e "$SMOKE_OUTPUT"

env \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  LIBERO_CONFIG_PATH="$BSP_WORK/cache/libero-config" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.task-ids 0 \
  --args.policy-variant baseline \
  --args.expected-action-horizon 10 \
  --args.num-trials-per-task 1 \
  --args.output-dir "$SMOKE_OUTPUT" \
  --args.config-name pi05_libero \
  --args.checkpoint-step 30000 \
  --args.code-sha "$CODE_SHA" \
  --args.dataset-revision v2.0 \
  --args.norm-hash "$OFFICIAL_NORM_HASH" \
  --args.checkpoint "$OFFICIAL_CHECKPOINT" \
  --args.container-digest ${HOST_RUNTIME_DIGEST} \
  --args.train-seed 42 \
  --args.eval-seed 42
```

成功条件不是“进程没有报错”，而是：`manifest.json`、`episodes.jsonl`、`summary.json`
存在；只有 1 条 episode；`incomplete_infrastructure_count=0`、`artifact_error_count=0`、
`acceptance_complete=true`。图像应是经过 180°校正的 RGB，不是倒置或黑屏。

## 官方四套件校准

服务端不变，把上面的 client 参数改为：

```text
--args.task-suite-name all
--args.num-trials-per-task 5
--args.output-dir /root/openpi-bsp-work/experiments/eval/official-h10-calibration-all-5
```

预期 4 suites × 10 tasks × 5 = 200 episodes。官方 checkpoint 成功率只用于校验环境链路，
不能当作本实验 baseline。

## phase-one checkpoint

先在服务端选择固定里程碑。下面示例是 baseline 1k；目录与 step 都由变量构造，无路径
占位符：

```bash
export STEP=1000
export CONFIG=pi05_libero_baseline_lora_h16
export EXP=phase1-short10k-seed42-baseline
export CHECKPOINT="$BSP_WORK/experiments/checkpoints/$CONFIG/$EXP/$STEP"
test "$(basename "$CHECKPOINT")" = "$STEP"
test -d "$CHECKPOINT/params"

cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  policy:checkpoint \
  --policy.config "$CONFIG" \
  --policy.dir "$CHECKPOINT"
```

baseline client 使用 horizon 16、执行前 8 步。BSP 则使用
`pi05_libero_bsp_lora_h16`、对应 BSP checkpoint、horizon 8，并额外传：

```text
--args.policy-variant baseline
--args.expected-action-horizon 16
--args.config-name pi05_libero_baseline_h16
```

LoRA checkpoint 的 `--args.config-name` 改为 `pi05_libero_baseline_lora_h16`；其余协议身份不变。
BSP evaluator 则使用 `--args.policy-variant bsp --args.expected-action-horizon 8`，因为服务端已经把
16 行 spline 参数解码成 8 个可执行动作。

```bash
export BSP_CACHE=/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/libero-v2.0-bsp-v2.npz
export BSP_CACHE_HASH="$(sha256sum "$BSP_CACHE" | awk '{print $1}')"
export BSP_CACHE_MANIFEST_FINGERPRINT=db8fe671f0e0ad33dcf2ef2e563c779c0f6c2cc4d91e314379d1c0bc64768213

# Add these exact arguments to the BSP evaluator command:
# --args.bsp-cache-hash "$BSP_CACHE_HASH"
# --args.bsp-cache-manifest-fingerprint "$BSP_CACHE_MANIFEST_FINGERPRINT"
```

正式 evaluator 命令、hash 派生和十个 run 顺序见 runbook 的“十次评测”。不得在 A/B 之间
改变 code SHA、dataset revision、runtime digest、train/eval seed、initial states 或 suite 顺序。

## 输出

每个 run 写：

```text
manifest.json
episodes.jsonl
tasks.csv
suites.csv
summary.json
artifact_errors.jsonl        only when an artifact failed
videos/<suite>/<task>/...    first success and first failure per task
```

simulator/network/reset/step 错误用相同 seed 最多重试两次；仍失败不混入成功率，并使 run
不完整。shape、NaN 或 BSP 解码错误是 policy failure。视频只做审计抽样，不等于所有 episode
都写视频。

十个完整 run 交给 `scripts/compare_libero_phase1.py`，固定比较 baseline/BSP 在
`0 / 1000 / 2000 / 5000 / 10000` 的结果。不要用目录名猜 variant/step；reporter 只信
manifest 身份。
