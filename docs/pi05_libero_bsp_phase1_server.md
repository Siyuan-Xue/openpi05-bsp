# π0.5 + LIBERO BSP 第一阶段服务器 Runbook

这是阿里云 DSW 单张 H20 上的唯一规范操作顺序。policy/training 使用 Python 3.11.9，
LIBERO simulator 使用 Python 3.8.20；二者通过 WebSocket 通信。本文不要求本地开发机安装
依赖、训练或运行仿真。

## 0. 冻结身份与实验协议

正在运行或等待验收的正式实验必须保持：

```text
tag:    phase1-runtime-2c09840
commit: 2c098404a3cce0c86f0b863dcd8d3aeb18a55d94
```

分支 `refactor/pi05-libero-bsp-slim` 是独立重构，不能 checkout 到活跃实验、不能替换正在
运行的进程，也不能用其结果补写冻结 runtime 的 manifest。等 baseline/BSP short10k、十次
评测和报告完成并明确授权后，瘦身分支才可单独执行服务器复验。

当前第一阶段只比较同一训练家族的：

```text
pi05_libero_baseline_lora_h16
pi05_libero_bsp_lora_h16
```

固定参数：`pi05_base` 初始化、LIBERO v2.0、seed 42、有效 batch 256、
`micro_batch_size=64`、EMA `None`、10,000 optimizer steps。永久里程碑为
`0 / 1000 / 2000 / 5000 / 10000`。A/B 严格串行，不挑最佳 checkpoint。full 配置保留供
未来硬件复验，不在当前 H20 上冒充已完成的正式协议。

停止规则：任一身份、数据、sidecar、norm、数值、checkpoint、EGL 或 artifact 门禁失败时，
停止后续阶段并保存日志；不自动降低 batch、不改 seed、不改训练家族、不覆盖旧产物。

## 1. 存储边界和禁忌

已确认的路径职责：

```text
/root/openpi-bsp-work/                         快速本地工作区；跨实例持久性未证明
  repo/openpi05-bsp/                           仓库
  venvs/                                       uv、CPython 和两个虚拟环境
  cache/                                       模型、包、JAX、EGL metadata
  staging/                                     归档前临时文件
  experiments/{assets,checkpoints,eval,logs,wandb}/

/mnt/data/siyuanxue/openpi-bsp/                本项目获准写入的数据命名空间
  data/lerobot/physical-intelligence/libero/   官方数据
  data/bspline-targets/                        BSP sidecar + verification
  experiments/{checkpoint-archives,eval-archives,assets-archives}/
                                                 完成写入后的持久归档
```

安全规则：

- `/mnt/data` 下只有 `/mnt/data/siyuanxue` 可写；禁止写、改名、移动或删除其他目录。
- 所有写命令使用完整绝对路径；进入 `/mnt/data` 后也不能直接在根目录创建文件。
- `/mnt/data` 是 `ossfs2` 对象存储，不把活跃 Orbax checkpoint 直接写到其上。文件锁、
  原子 rename 和随机 I/O 语义尚未证明。
- `/root` 快但不保证实例重建后保留。正式训练前至少保留 80 GiB 空间；每个固定 milestone
  必须等待 checkpoint manager 完成后再校验和归档。
- `/mnt/workspace` 约 30 GiB，只作终端导航或小文件，不放模型、数据、checkpoint。
- 不污染系统 Python，不全局 `pip install`，不替换驱动、系统 CUDA 或基础镜像 PyTorch。
- 不执行 `git reset --hard`、`git clean`、未限定通配符删除，或删除正在增长的 `.partial`、
  Orbax 临时目录和未核实实验目录。
- 不转换 LIBERO v2.0，不使用 `modified_libero_rlds`，不初始化 ALOHA 子模块。

每次写入数据盘前用只读门禁：

```bash
TARGET=/mnt/data/siyuanxue/openpi-bsp
RESOLVED="$(realpath -m "$TARGET")"
case "$RESOLVED" in
  /mnt/data/siyuanxue/*) printf 'write_namespace=PASS %s\n' "$RESOLVED" ;;
  *) printf 'write_namespace=FAIL %s\n' "$RESOLVED" >&2; exit 2 ;;
esac
```

## 2. 每个新 shell 的变量

这些 `export` 只作用于当前 shell；除非单独审计，不把“曾经 export”写成永久配置。

```bash
set -euo pipefail

export BSP_WORK=/root/openpi-bsp-work
export BSP_REPO_DIR="$BSP_WORK/repo/openpi05-bsp"
export OPENPI_VENV="$BSP_WORK/venvs/openpi"
export OPENPI_PY="$OPENPI_VENV/bin/python"
export LIBERO_VENV="$BSP_WORK/venvs/libero-py38"
export LIBERO_PY="$LIBERO_VENV/bin/python"
export UV_VERSION=0.11.32
export UV_BIN_DIR="$BSP_WORK/venvs/uv-$UV_VERSION-bin"
export UV_BIN="$UV_BIN_DIR/uv"
export UV_ARCHIVE_NAME=uv-x86_64-unknown-linux-gnu.tar.gz
export UV_ARCHIVE="$BSP_WORK/staging/$UV_ARCHIVE_NAME"
export UV_CHECKSUM_FILE="$UV_ARCHIVE.sha256"
export UV_RELEASE_BASE="https://releases.astral.sh/github/uv/releases/download/$UV_VERSION"

export OPENPI_DATA_HOME="$BSP_WORK/cache/openpi"
export HF_HOME="$BSP_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$BSP_WORK/cache/uv"
export JAX_COMPILATION_CACHE_DIR="$BSP_WORK/cache/jax"
export WANDB_DIR="$BSP_WORK/experiments/wandb"
export WANDB_MODE=offline
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

export PERSIST_ROOT=/mnt/data/siyuanxue/openpi-bsp
export LIBERO_DATASET_DIR="$PERSIST_ROOT/data/lerobot/physical-intelligence/libero"
export BSP_CACHE="$PERSIST_ROOT/data/bspline-targets/libero-v2.0-bsp-v2.npz"
export BSP_VERIFY="$PERSIST_ROOT/data/bspline-targets/libero-v2.0-bsp-v2.verification.json"

export ASSETS_BASE="$BSP_WORK/experiments/assets"
export CHECKPOINT_BASE="$BSP_WORK/experiments/checkpoints"
export EVAL_BASE="$BSP_WORK/experiments/eval"
export LOG_BASE="$BSP_WORK/experiments/logs"
export EGL_VENDOR_JSON="$BSP_WORK/cache/egl/10_nvidia.json"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

只读核验：

```bash
test "$(realpath -m "$PERSIST_ROOT")" = /mnt/data/siyuanxue/openpi-bsp
findmnt -T /mnt/data -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,OPTIONS
df -hT /root /mnt/workspace /mnt/data
git -C "$BSP_REPO_DIR" status --short --branch
git -C "$BSP_REPO_DIR" rev-parse HEAD
git -C "$BSP_REPO_DIR" submodule status third_party/libero
"$OPENPI_PY" -c 'import platform,scipy,jax; print(platform.python_version()); print(scipy.__version__); print(jax.devices())'
"$LIBERO_PY" -c 'import platform; print(platform.python_version())'
```

## 3. 源码与锁定环境

新服务器只 clone fork，并只初始化固定 LIBERO gitlink：

```bash
mkdir -p "$BSP_WORK/repo" "$BSP_WORK/venvs" "$BSP_WORK/cache" "$BSP_WORK/experiments/logs"
git clone --branch main --single-branch \
  https://github.com/Siyuan-Xue/openpi05-bsp.git \
  "$BSP_REPO_DIR"
git -C "$BSP_REPO_DIR" checkout --detach phase1-runtime-2c09840
test "$(git -C "$BSP_REPO_DIR" rev-parse HEAD)" = \
  2c098404a3cce0c86f0b863dcd8d3aeb18a55d94
git -C "$BSP_REPO_DIR" -c http.version=HTTP/1.1 \
  submodule update --init third_party/libero
test "$(git -C "$BSP_REPO_DIR/third_party/libero" rev-parse HEAD)" = \
  f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

若 GitHub 偶发 TLS 中断，只对同一只读 fetch/pull 使用
`-c http.version=HTTP/1.1` 重试；不能切换到未知 mirror 或浮动子模块 HEAD。

环境版本是：uv 0.11.32、OpenPI CPython 3.11.9、LIBERO CPython 3.8.20、
SciPy 1.15.3。uv 是后续所有环境命令的显式 bootstrap 前提。下面直接下载 Astral
[immutable 0.11.32 release](https://github.com/astral-sh/uv/releases/tag/0.11.32) 的 x86-64
Linux 归档和官方 checksum；同时把官方 checksum 的预期值固定在 runbook 中，避免只信任同一次
下载。先在独立 staging 目录核验，再只抽取 `uv` 到项目目录；不执行远程安装脚本，也不修改
shell profile 或系统 Python：

```bash
export EXPECTED_UV_SHA256=aab924fd522efd06f1c5f3b93a243864fc453132c94b2dc49f1371b528a4b967

test "$UV_VERSION" = 0.11.32
test "$UV_BIN_DIR" = /root/openpi-bsp-work/venvs/uv-0.11.32-bin
test "$UV_ARCHIVE" = \
  /root/openpi-bsp-work/staging/uv-x86_64-unknown-linux-gnu.tar.gz
test ! -e "$UV_BIN_DIR"
test ! -e "$UV_ARCHIVE"
test ! -e "$UV_CHECKSUM_FILE"
mkdir -p "$BSP_WORK/staging" "$BSP_WORK/venvs" "$LOG_BASE"

curl --proto '=https' --tlsv1.2 -fL \
  --connect-timeout 15 --max-time 600 \
  "$UV_RELEASE_BASE/$UV_ARCHIVE_NAME" \
  -o "$UV_ARCHIVE"
curl --proto '=https' --tlsv1.2 -fL \
  --connect-timeout 15 --max-time 60 \
  "$UV_RELEASE_BASE/$UV_ARCHIVE_NAME.sha256" \
  -o "$UV_CHECKSUM_FILE"

test "$(awk '{print $1}' "$UV_CHECKSUM_FILE")" = "$EXPECTED_UV_SHA256"
test "$(awk '{print $2}' "$UV_CHECKSUM_FILE")" = "$UV_ARCHIVE_NAME"
(
  cd "$(dirname "$UV_ARCHIVE")"
  sha256sum -c "$(basename "$UV_CHECKSUM_FILE")"
)
printf '%s  %s\n' "$EXPECTED_UV_SHA256" "$UV_ARCHIVE_NAME" \
  | tee "$LOG_BASE/uv-$UV_VERSION-archive.sha256"

mkdir "$UV_BIN_DIR"
tar -xzf "$UV_ARCHIVE" \
  -C "$UV_BIN_DIR" \
  --strip-components=1 \
  uv-x86_64-unknown-linux-gnu/uv

test -x "$UV_BIN"
test "$("$UV_BIN" --version | awk '{print $2}')" = "$UV_VERSION"
printf 'uv_bootstrap=PASS %s\n' "$("$UV_BIN" --version)"
```

若组织网络禁止 `releases.astral.sh`，先在可信联网机器下载这两个同名文件并核验上述硬编码
SHA256，再原样上传到 `$BSP_WORK/staging`，从 checksum 校验块继续；不要改用浮动 `latest`、
未知镜像或全局 `pip install`。若 `$UV_BIN_DIR` 已存在，则跳过创建/抽取，只执行最后三行版本
门禁；路径中出现半成品时停止人工检查，不覆盖。

首次创建两个 Python 环境时：

```bash
export UV_PYTHON_INSTALL_DIR="$BSP_WORK/venvs/python-builds"

env UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
  "$UV_BIN" python install --no-bin 3.11.9
export OPENPI_PY_SOURCE="$(env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
  "$UV_BIN" python find 3.11.9)"

cd "$BSP_REPO_DIR"
env UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
  UV_PROJECT_ENVIRONMENT="$OPENPI_VENV" UV_HTTP_TIMEOUT=120 \
  "$UV_BIN" sync --frozen --python "$OPENPI_PY_SOURCE"

env UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
  "$UV_BIN" python install --no-bin 3.8
export LIBERO_PY_SOURCE="$(env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
  "$UV_BIN" python find 3.8)"
env UV_CACHE_DIR="$UV_CACHE_DIR" "$UV_BIN" venv \
  --python "$LIBERO_PY_SOURCE" "$LIBERO_VENV"

"$UV_BIN" pip install --python "$LIBERO_PY" \
  -r examples/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113
"$UV_BIN" pip install --python "$LIBERO_PY" \
  -r third_party/libero/requirements.txt
"$UV_BIN" pip install --python "$LIBERO_PY" --no-deps \
  -e packages/openpi-client -e third_party/libero
```

不要复用基础镜像 Python 3.12。`libero` editable distribution 的源码包位于子模块内层；
运行 evaluator 时仍显式设置 `PYTHONPATH`，避免不同 packaging layout 造成歧义。

## 4. 轻量代码门禁

先确认 help，再运行不需要数据或 GPU 训练的合同。瘦身分支增加了文档合同；冻结 runtime
只运行该 tag 实际存在的测试，不从新分支复制测试文件过去。

```bash
cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/train.py pi05_libero_baseline_lora_h16 --help
"$OPENPI_PY" scripts/train.py pi05_libero_bsp_lora_h16 --help
"$OPENPI_PY" scripts/prepare_libero_bsp.py --help
"$OPENPI_PY" scripts/compute_norm_stats.py --help
"$OPENPI_PY" scripts/serve_policy.py --help
"$LIBERO_PY" examples/libero/main.py --help

"$OPENPI_PY" -m pytest -q \
  src/openpi/training/bsp_test.py \
  src/openpi/training/bsp_dataset_test.py \
  src/openpi/training/data_loader_test.py \
  src/openpi/policies/libero_policy_test.py \
  scripts/prepare_libero_bsp_test.py \
  scripts/compute_norm_stats_test.py \
  scripts/train_test.py \
  scripts/libero_eval_test.py \
  scripts/compare_libero_phase1_test.py
```

warning 不是通过证据；必须检查返回码和最终计数。第三方 deprecation warning 可记录，但
Traceback、OOM、non-finite 或失败断言必须停止。

## 5. 官方数据和模型资产

只下载 `physical-intelligence/libero@v2.0`：

```bash
mkdir -p "$PERSIST_ROOT/data/lerobot/physical-intelligence" \
  "$PERSIST_ROOT/data/bspline-targets" "$OPENPI_DATA_HOME" "$LOG_BASE"
cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/prepare_libero_bsp.py \
  --mode download \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --repo-id physical-intelligence/libero \
  --revision v2.0 \
  --action-key actions \
  2>&1 | tee "$LOG_BASE/libero-v2-download.log"
```

完成门禁必须显示：1,693 episodes、273,465 frames、40 tasks、10 Hz。LeRobot 关于 v2.1
转换的提示只记录，不执行转换。

预取 base、官方校准 checkpoint 和 tokenizer：

```bash
cd "$BSP_REPO_DIR"
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" "$OPENPI_PY" - <<'PY'
from openpi.shared.download import maybe_download

for uri in (
    "gs://openpi-assets/checkpoints/pi05_base",
    "gs://openpi-assets/checkpoints/pi05_libero",
):
    print(uri, "->", maybe_download(uri))
print(
    "tokenizer ->",
    maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}),
)
PY

test -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"
test -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/params"
test -s "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
```

`.partial` 正在增长代表下载未完成；后台进程退出也不代表完成，必须同时检查日志终态和
最终 `params` 目录。

## 6. Headless EGL

DSW 缺失系统 GLVND vendor JSON 时，在项目 cache 中提供最小 NVIDIA ICD，不修改系统目录：

```bash
mkdir -p "$(dirname "$EGL_VENDOR_JSON")"
printf '%s\n' \
  '{' \
  '  "file_format_version": "1.0.0",' \
  '  "ICD": {"library_path": "libEGL_nvidia.so.0"}' \
  '}' > "$EGL_VENDOR_JSON"
"$LIBERO_PY" -m json.tool "$EGL_VENDOR_JSON"

env \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" -c \
  'import mujoco; c=mujoco.GLContext(16,16); c.make_current(); c.free(); print("mujoco_egl=PASS")'
```

LIBERO 环境变量固定为：

```bash
export LIBERO_PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero:$BSP_REPO_DIR/third_party/libero/libero"
export LIBERO_CONFIG_PATH="$BSP_WORK/cache/libero-config"
mkdir -p "$LIBERO_CONFIG_PATH"
```

先做真实 task 0 × 1 reset/render/step 冒烟。完整脚本入口和参数见
[LIBERO README](../examples/libero/README.md)。成功必须包含两张 `(256, 256, 3)` uint8 图像、
有限值、一次 dummy action step 和退出码 0。缺少数据集软链接的 warning 不等于失败；EGL
ImportError、黑屏、错误 shape 或崩溃是失败。

## 7. 官方 checkpoint 校准

先生成 host runtime 身份。代码字段仍名为 `container_digest`，在双环境路线中保存的是两个
解释器、LIBERO gitlink 和 EGL 配置的可复算摘要，不是镜像 ID：

```bash
export CODE_SHA="$(git -C "$BSP_REPO_DIR" rev-parse HEAD)"
export HOST_RUNTIME_DIGEST="sha256:$(
  {
    "$OPENPI_PY" --version
    "$LIBERO_PY" --version
    git -C "$BSP_REPO_DIR/third_party/libero" rev-parse HEAD
    sha256sum "$EGL_VENDOR_JSON"
  } | sha256sum | awk '{print $1}'
)"
```

终端 A 启动官方服务：

```bash
cd "$BSP_REPO_DIR"
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR" \
"$OPENPI_PY" scripts/serve_policy.py \
  --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
```

终端 B 先跑一个 task，再跑四套件 × 10 tasks × 5 trials = 200 episodes：

```bash
export OFFICIAL_NORM="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
export OFFICIAL_NORM_HASH="$(sha256sum "$OFFICIAL_NORM" | awk '{print $1}')"
export OFFICIAL_CAL="$EVAL_BASE/official-h10-calibration-all-5"
test ! -e "$OFFICIAL_CAL"

cd "$BSP_REPO_DIR"
env \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name all \
  --args.policy-variant baseline \
  --args.expected-action-horizon 10 \
  --args.num-trials-per-task 5 \
  --args.output-dir "$OFFICIAL_CAL" \
  --args.config-name pi05_libero \
  --args.checkpoint-step 30000 \
  --args.code-sha "$CODE_SHA" \
  --args.dataset-revision v2.0 \
  --args.norm-hash "$OFFICIAL_NORM_HASH" \
  --args.checkpoint "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" \
  --args.container-digest "$HOST_RUNTIME_DIGEST" \
  --args.train-seed 42 \
  --args.eval-seed 42
```

校准不是正式 A/B 结果。门禁是 `episodes.jsonl=200`、四套件齐全、无 infrastructure/artifact
error、`acceptance_complete=true`；成功率只用于发现图像/动作/环境链路错误。

## 8. BSP sidecar build 和 full verify

sidecar 只含 spline target、frame mapping 和 manifest，不复制 observation 或图像：

```bash
export LOCAL_BSP_DIR="$BSP_WORK/staging/bspline-targets"
export LOCAL_BSP_CACHE="$LOCAL_BSP_DIR/libero-v2.0-bsp-v2.npz"
export LOCAL_BSP_VERIFY="$LOCAL_BSP_DIR/libero-v2.0-bsp-v2.verification.json"

test "$(realpath -m "$LOCAL_BSP_DIR")" = /root/openpi-bsp-work/staging/bspline-targets
test "$(realpath -m "$(dirname "$BSP_CACHE")")" = \
  /mnt/data/siyuanxue/openpi-bsp/data/bspline-targets
mkdir -p "$LOCAL_BSP_DIR" "$(dirname "$BSP_CACHE")"
test ! -e "$LOCAL_BSP_CACHE"
test ! -e "$LOCAL_BSP_VERIFY"
test ! -e "$BSP_CACHE"
test ! -e "$BSP_VERIFY"

cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/prepare_libero_bsp.py \
  --mode build \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --cache-path "$LOCAL_BSP_CACHE" \
  --repo-id physical-intelligence/libero \
  --revision v2.0 \
  --action-key actions \
  2>&1 | tee "$LOG_BASE/bsp-build.log"

"$OPENPI_PY" scripts/prepare_libero_bsp.py \
  --mode verify \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --cache-path "$LOCAL_BSP_CACHE" \
  --diagnostics-path "$LOCAL_BSP_VERIFY" \
  --repo-id physical-intelligence/libero \
  --revision v2.0 \
  --action-key actions \
  2>&1 | tee "$LOG_BASE/bsp-verify.log"

export LOCAL_BSP_HASH="$(sha256sum "$LOCAL_BSP_CACHE" | awk '{print $1}')"
"$OPENPI_PY" - "$LOCAL_BSP_VERIFY" <<'PY'
import json
import sys

diagnostics = json.load(open(sys.argv[1], encoding="utf-8"))
assert diagnostics["verification_passed"] is True
print("bsp_full_verify_gate=PASS")
PY

cp --no-clobber "$LOCAL_BSP_CACHE" "$BSP_CACHE"
cp --no-clobber "$LOCAL_BSP_VERIFY" "$BSP_VERIFY"
test "$LOCAL_BSP_HASH" = "$(sha256sum "$BSP_CACHE" | awk '{print $1}')"
test "$(sha256sum "$LOCAL_BSP_VERIFY" | awk '{print $1}')" = \
  "$(sha256sum "$BSP_VERIFY" | awk '{print $1}')"
```

预期结构是 targets `(259121, 16, 8)` float32、mapping `(273465,)` uint32，manifest
fingerprint 为 `db8fe671f0e0ad33dcf2ef2e563c779c0f6c2cc4d91e314379d1c0bc64768213`。
`verification_passed` 必须为 true；最大重建误差、knot 单调性、tail padding、episode 边界、
全 frame coverage、重建一致性和缓存确定性必须逐项通过。

build/verify 必须在 `/root` staging 完成，避免把临时文件、锁和原子 rename 交给未验证的
对象存储语义。只有完整验证后才以 no-clobber 方式复制两个稳定文件，并逐字节校验目标。
训练可只读使用持久 `$BSP_CACHE`。

## 9. A/B normalization stats

详见 [norm stats 文档](norm_stats.md)。固定输出目录不同，但 state 输入相同：

```bash
export BASELINE_NORM_DIR="$ASSETS_BASE/pi05_libero_baseline_h16/libero_baseline_h16"
export BSP_NORM_DIR="$ASSETS_BASE/pi05_libero_bsp_h16/libero_bsp_h16"
export BASELINE_LORA_NORM_DIR="$ASSETS_BASE/pi05_libero_baseline_lora_h16/libero_baseline_h16"
export BSP_LORA_NORM_DIR="$ASSETS_BASE/pi05_libero_bsp_lora_h16/libero_bsp_h16"
export NORM_COMPARISON="$ASSETS_BASE/libero-phase1-norm-comparison.json"

cd "$BSP_REPO_DIR"
"$OPENPI_PY" scripts/compute_norm_stats.py \
  pi05_libero_baseline_h16 \
  --assets-dir "$ASSETS_BASE/pi05_libero_baseline_h16" \
  --dataset-root "$LIBERO_DATASET_DIR"

"$OPENPI_PY" scripts/compute_norm_stats.py \
  pi05_libero_bsp_h16 \
  --assets-dir "$ASSETS_BASE/pi05_libero_bsp_h16" \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --bsp-cache-path "$BSP_CACHE" \
  --compare-state-stats-with "$BASELINE_NORM_DIR" \
  --norm-comparison-output "$NORM_COMPARISON"

mkdir -p "$BASELINE_LORA_NORM_DIR" "$BSP_LORA_NORM_DIR"
test ! -e "$BASELINE_LORA_NORM_DIR/norm_stats.json"
test ! -e "$BSP_LORA_NORM_DIR/norm_stats.json"
test ! -e "$BASELINE_LORA_NORM_DIR/.norm_stats.json.publish-tmp"
test ! -e "$BSP_LORA_NORM_DIR/.norm_stats.json.publish-tmp"
cp "$BASELINE_NORM_DIR/norm_stats.json" \
  "$BASELINE_LORA_NORM_DIR/.norm_stats.json.publish-tmp"
mv "$BASELINE_LORA_NORM_DIR/.norm_stats.json.publish-tmp" \
  "$BASELINE_LORA_NORM_DIR/norm_stats.json"
cp "$BSP_NORM_DIR/norm_stats.json" \
  "$BSP_LORA_NORM_DIR/.norm_stats.json.publish-tmp"
mv "$BSP_LORA_NORM_DIR/.norm_stats.json.publish-tmp" \
  "$BSP_LORA_NORM_DIR/norm_stats.json"
test "$(sha256sum "$BASELINE_NORM_DIR/norm_stats.json" | awk '{print $1}')" = \
  "$(sha256sum "$BASELINE_LORA_NORM_DIR/norm_stats.json" | awk '{print $1}')"
test "$(sha256sum "$BSP_NORM_DIR/norm_stats.json" | awk '{print $1}')" = \
  "$(sha256sum "$BSP_LORA_NORM_DIR/norm_stats.json" | awk '{print $1}')"
```

门禁：两个 `norm_stats.json` 存在且有限；state 的 `mean/std/q01/q99` 数值一致；action
hash 不同；BSP action 是 8D，knot `q01 < q99`。LoRA 配置复用同一 variant 的资产，不重新
混算一套统计，但必须原子发布到各自配置名对应的 asset 根；只保留 full 路径会使 LoRA
训练找不到 stats。

## 10. Pilot 与正式训练

正式训练前 A/B 各完成 100 optimizer-step LoRA pilot。一次只允许一个 GPU 进程；路径必须
是新目录；日志不得有 OOM、RESOURCE_EXHAUSTED、Traceback、NaN 或 Inf。pilot 与正式命令
保持有效 batch 256、micro-batch 64、EMA None、seed 42。

正式 baseline：

```bash
export BASELINE_CONFIG=pi05_libero_baseline_lora_h16
export BSP_CONFIG=pi05_libero_bsp_lora_h16
export BASELINE_EXP=phase1-short10k-seed42-baseline
export BSP_EXP=phase1-short10k-seed42-bsp
export BASELINE_RUN="$CHECKPOINT_BASE/$BASELINE_CONFIG/$BASELINE_EXP"
export BSP_RUN="$CHECKPOINT_BASE/$BSP_CONFIG/$BSP_EXP"

test ! -e "$BASELINE_RUN"
test ! -e "$BSP_RUN"
test "$(df --output=avail -B1 /root | tail -n 1)" -ge $((80 * 1024**3))

cd "$BSP_REPO_DIR"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
WANDB_MODE=offline \
"$OPENPI_PY" scripts/train.py "$BASELINE_CONFIG" \
  --exp-name "$BASELINE_EXP" \
  --seed 42 \
  --batch-size 256 \
  --micro-batch-size 64 \
  --num-train-steps 10000 \
  --save-interval 1000 \
  --keep-period 10000 \
  --ema-decay None \
  --assets-base-dir "$ASSETS_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_BASE" \
  --data.lerobot-root "$LIBERO_DATASET_DIR" \
  2>&1 | tee "$LOG_BASE/train-phase1-short10k-seed42-baseline.log"
```

baseline 完成并通过全部门禁后才运行 BSP；命令只多配置和 sidecar：

```bash
test ! -e "$BSP_RUN"
test "$(df --output=avail -B1 /root | tail -n 1)" -ge $((80 * 1024**3))

cd "$BSP_REPO_DIR"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
WANDB_MODE=offline \
"$OPENPI_PY" scripts/train.py "$BSP_CONFIG" \
  --exp-name "$BSP_EXP" \
  --seed 42 \
  --batch-size 256 \
  --micro-batch-size 64 \
  --num-train-steps 10000 \
  --save-interval 1000 \
  --keep-period 10000 \
  --ema-decay None \
  --assets-base-dir "$ASSETS_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_BASE" \
  --data.lerobot-root "$LIBERO_DATASET_DIR" \
  --data.bsp-cache-path "$BSP_CACHE" \
  2>&1 | tee "$LOG_BASE/train-phase1-short10k-seed42-bsp.log"
```

不加 `--overwrite`。恢复前确认最近目录位于完整 optimizer-step 边界，再用同一命令加
`--resume`。每个 variant 都必须存在：

```bash
for root in "$BASELINE_RUN" "$BSP_RUN"; do
  for step in 0 1000 2000 5000 10000; do
    test -d "$root/$step/params"
    test -d "$root/$step/train_state"
    test -d "$root/$step/assets"
  done
  test -z "$(find "$root" -type d -name '*.orbax-checkpoint-tmp-*' -print -quit)"
done
```

step 0 必须在第一次正梯度更新前完成；它是各自 config/norm/protocol 的 checkpoint，不能用
官方 `pi05_libero` checkpoint 代替。

## 11. 十次评测

每个 checkpoint 运行四套件 × 10 tasks × 50 initial states = 2,000 episodes；十个 run
总计 20,000。baseline 服务输出 h16，执行前 8 步；BSP 服务在反归一化后解码 h8。两者
都在 10 Hz 下每 8 步重规划，并从 `(suite, task, init_state, replan_index)` 派生相同 flow
noise seed。

对每个 `variant/step`：

1. 在 OpenPI 终端用 `scripts/serve_policy.py policy:checkpoint` 加载唯一 checkpoint；
2. 在 LIBERO 终端以 `--args.task-suite-name all --args.num-trials-per-task 50` 运行 evaluator；
3. baseline 使用 `--args.expected-action-horizon 16`，两个 BSP cache 身份留空；
4. BSP 使用 `--args.expected-action-horizon 8`，显式传 NPZ SHA256 和 manifest fingerprint；
5. `--args.checkpoint-step`、路径末级、config、norm hash 必须对应；
6. 每次输出必须是新目录 `$EVAL_BASE/{variant}-step-{step}`。

通用 BSP evaluator 示例：

```bash
export STEP=1000
export CHECKPOINT="$BSP_RUN/$STEP"
export OUTPUT="$EVAL_BASE/bsp-step-$STEP"
export BSP_CACHE_HASH="$(sha256sum "$BSP_CACHE" | awk '{print $1}')"
export BSP_FINGERPRINT=db8fe671f0e0ad33dcf2ef2e563c779c0f6c2cc4d91e314379d1c0bc64768213
export BSP_NORM_HASH="$(sha256sum "$CHECKPOINT/assets/libero_bsp_h16/norm_stats.json" | awk '{print $1}')"
test ! -e "$OUTPUT"

env \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name all \
  --args.policy-variant bsp \
  --args.expected-action-horizon 8 \
  --args.num-trials-per-task 50 \
  --args.output-dir "$OUTPUT" \
  --args.config-name "$BSP_CONFIG" \
  --args.checkpoint-step "$STEP" \
  --args.code-sha "$CODE_SHA" \
  --args.dataset-revision v2.0 \
  --args.bsp-cache-hash "$BSP_CACHE_HASH" \
  --args.bsp-cache-manifest-fingerprint "$BSP_FINGERPRINT" \
  --args.norm-hash "$BSP_NORM_HASH" \
  --args.checkpoint "$CHECKPOINT" \
  --args.container-digest "$HOST_RUNTIME_DIGEST" \
  --args.train-seed 42 \
  --args.eval-seed 42
```

基础设施错误按原 seed 最多重试两次；仍失败则 `acceptance_complete=false`，不混入成功率。
非法 shape、NaN 或无效 BSP 区间是策略失败。每个 task 只保留首个成功和首个失败视频，
但视频编码失败仍使 run 不完整。

## 12. 配对报告

十次评测全部完整后：

```bash
export REPORT_DIR="$EVAL_BASE/phase1-comparison-seed42"
test ! -e "$REPORT_DIR"

cd "$BSP_REPO_DIR"
PYTHONPATH="$BSP_REPO_DIR/packages/openpi-client/src" \
"$OPENPI_PY" scripts/compare_libero_phase1.py \
  "$EVAL_BASE/baseline-step-0" \
  "$EVAL_BASE/bsp-step-0" \
  "$EVAL_BASE/baseline-step-1000" \
  "$EVAL_BASE/bsp-step-1000" \
  "$EVAL_BASE/baseline-step-2000" \
  "$EVAL_BASE/bsp-step-2000" \
  "$EVAL_BASE/baseline-step-5000" \
  "$EVAL_BASE/bsp-step-5000" \
  "$EVAL_BASE/baseline-step-10000" \
  "$EVAL_BASE/bsp-step-10000" \
  --bsp-verification "$BSP_VERIFY" \
  --norm-comparison "$NORM_COMPARISON" \
  --output-dir "$REPORT_DIR"
```

比较器只接受同一 full 或同一 LoRA 家族、恰好五个里程碑、完全配对的 20,000 episodes。
它固定生成：`task_comparison.csv`、`suite_comparison.csv`、`learning_curve.csv`、
`comparison.json`、`report.md`、`learning_curve.svg`。主指标是四套件分层宏平均成功率；
BSP-baseline 差值使用 seed 42、10,000 次 task-stratified paired bootstrap 给出 95% CI。

## 13. 持久归档

活跃 checkpoint、stats 和评测先保持在 `/root`。只有训练/服务/evaluator 都退出、checkpoint
manager 完成、无临时目录，且十次 run 与报告门禁通过后，才归档稳定树。必须归档三组不可替代
产物，而不只是 baseline checkpoint：

1. baseline 与 BSP 两个 short10k checkpoint tree；
2. baseline/BSP full 与 LoRA norm、comparison JSON；
3. 官方校准、十次 phase-one evaluation（包括抽样视频）和最终 report。

先建立三个本地 tar。每个 member 都必须存在；tar 仍在 `/root` 构建，不能让对象存储承担活跃
小文件读取和临时写入：

```bash
export CHECKPOINT_ARCHIVE_DIR="$PERSIST_ROOT/experiments/checkpoint-archives"
export ASSET_ARCHIVE_DIR="$PERSIST_ROOT/experiments/assets-archives"
export EVAL_ARCHIVE_DIR="$PERSIST_ROOT/experiments/eval-archives"

test "$(realpath -m "$CHECKPOINT_ARCHIVE_DIR")" = \
  /mnt/data/siyuanxue/openpi-bsp/experiments/checkpoint-archives
test "$(realpath -m "$ASSET_ARCHIVE_DIR")" = \
  /mnt/data/siyuanxue/openpi-bsp/experiments/assets-archives
test "$(realpath -m "$EVAL_ARCHIVE_DIR")" = \
  /mnt/data/siyuanxue/openpi-bsp/experiments/eval-archives
mkdir -p "$BSP_WORK/staging" \
  "$CHECKPOINT_ARCHIVE_DIR" "$ASSET_ARCHIVE_DIR" "$EVAL_ARCHIVE_DIR"

export LOCAL_CHECKPOINT_ARCHIVE="$BSP_WORK/staging/phase1-short10k-seed42-checkpoints.tar"
export LOCAL_ASSET_ARCHIVE="$BSP_WORK/staging/phase1-short10k-seed42-norm-assets.tar"
export LOCAL_EVAL_ARCHIVE="$BSP_WORK/staging/phase1-short10k-seed42-evaluations.tar"
for archive in "$LOCAL_CHECKPOINT_ARCHIVE" "$LOCAL_ASSET_ARCHIVE" "$LOCAL_EVAL_ARCHIVE"; do
  test ! -e "$archive"
  test ! -e "$archive.sha256"
done

CHECKPOINT_MEMBERS=(
  "$BASELINE_CONFIG/$BASELINE_EXP"
  "$BSP_CONFIG/$BSP_EXP"
)
for member in "${CHECKPOINT_MEMBERS[@]}"; do
  test -d "$CHECKPOINT_BASE/$member"
done
tar -cf "$LOCAL_CHECKPOINT_ARCHIVE" \
  -C "$CHECKPOINT_BASE" "${CHECKPOINT_MEMBERS[@]}"

ASSET_MEMBERS=(
  pi05_libero_baseline_h16
  pi05_libero_bsp_h16
  pi05_libero_baseline_lora_h16
  pi05_libero_bsp_lora_h16
  libero-phase1-norm-comparison.json
)
for member in "${ASSET_MEMBERS[@]}"; do
  test -e "$ASSETS_BASE/$member"
done
tar -cf "$LOCAL_ASSET_ARCHIVE" \
  -C "$ASSETS_BASE" "${ASSET_MEMBERS[@]}"

EVAL_MEMBERS=(
  official-h10-calibration-all-5
  baseline-step-0 bsp-step-0
  baseline-step-1000 bsp-step-1000
  baseline-step-2000 bsp-step-2000
  baseline-step-5000 bsp-step-5000
  baseline-step-10000 bsp-step-10000
  phase1-comparison-seed42
)
for member in "${EVAL_MEMBERS[@]}"; do
  test -d "$EVAL_BASE/$member"
done
tar -cf "$LOCAL_EVAL_ARCHIVE" \
  -C "$EVAL_BASE" "${EVAL_MEMBERS[@]}"
```

再用同一个 no-clobber + SHA256 门禁复制三组归档。checksum 文件写入持久 archive 的 basename，
所以最终 `sha256sum -c` 实际读取持久目标，而不是误验 `/root` 源文件：

```bash
persist_verified_archive() {
  local local_archive="$1"
  local archive_dir="$2"
  local archive_name
  local local_checksum
  local persistent_archive
  local persistent_checksum
  local source_hash
  local destination_hash

  archive_name="$(basename "$local_archive")"
  local_checksum="$local_archive.sha256"
  persistent_archive="$archive_dir/$archive_name"
  persistent_checksum="$archive_dir/$archive_name.sha256"

  test -s "$local_archive"
  test ! -e "$local_checksum"
  test ! -e "$persistent_archive"
  test ! -e "$persistent_checksum"

  source_hash="$(sha256sum "$local_archive" | awk '{print $1}')"
  printf '%s  %s\n' "$source_hash" "$archive_name" > "$local_checksum"
  (
    cd "$(dirname "$local_archive")"
    sha256sum -c "$(basename "$local_checksum")"
  )

  cp --no-clobber "$local_archive" "$persistent_archive"
  destination_hash="$(sha256sum "$persistent_archive" | awk '{print $1}')"
  test "$source_hash" = "$destination_hash"
  cp --no-clobber "$local_checksum" "$persistent_checksum"
  (
    cd "$archive_dir"
    sha256sum -c "$(basename "$persistent_checksum")"
  )
}

persist_verified_archive "$LOCAL_CHECKPOINT_ARCHIVE" "$CHECKPOINT_ARCHIVE_DIR"
persist_verified_archive "$LOCAL_ASSET_ARCHIVE" "$ASSET_ARCHIVE_DIR"
persist_verified_archive "$LOCAL_EVAL_ARCHIVE" "$EVAL_ARCHIVE_DIR"
```

三个 archive 及三个 `.sha256` 全部通过后，持久归档门禁才完成。归档后也不自动删除本地
checkpoint、stats、评测或视频。空间不足时先停止新作业并人工审计；不要用更强的删除命令处理
锁定文件或不明目录。

## 14. 最终验收清单

- runtime commit/tag、LIBERO gitlink、数据 revision、sidecar hash/fingerprint、norm hashes 全部记录；
- official h10 task-0 冒烟和 200-episode 校准完整；
- sidecar full verify 和 A/B norm gate 完整；
- A/B pilot 使用同一协议且数值有限；
- A/B 各有 0/1k/2k/5k/10k 的 params、train_state、assets；
- 十个 run 各 2,000 episodes，无 infrastructure/artifact error；
- 比较器读取原始 diagnostics 并只生成六个固定报告文件；
- 训练、评测、归档全程没有写入 `/mnt/data` 下未获授权的位置；
- slim 分支仍未替换冻结实验；其服务器门禁被明确记录为“尚待独立执行”。

只有全部为真，第一阶段才完成。BSP 不被预设必须优于 baseline；可审计负结果同样有效。
