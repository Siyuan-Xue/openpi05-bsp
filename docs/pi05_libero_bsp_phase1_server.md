# π0.5 + LIBERO BSP 第一阶段服务器 Runbook

本文是阿里云单卡 H20 服务器从空机到第一阶段验收的唯一执行顺序。所有依赖、数据、训练、推理和仿真命令都只在服务器执行，不在本地 Mac 执行。

第一阶段只比较同一 `pi05_base`、同一 LIBERO v2.0 数据和同一 seed 42 下同一训练家族的 Baseline 与 BSP。当前单卡 H20 路线使用 `pi05_libero_baseline_lora_h16` 与 `pi05_libero_bsp_lora_h16`；报告器也接受完整的全量微调家族，但禁止混合两个家族。不做 2×/4× 加速、segment alignment、异步执行、gripper 阈值化或额外 loss，也不从五个里程碑中挑选“最佳 checkpoint”。BSP 不被预设为必须优于 baseline；可审计的负结果同样是有效结果。

## 0. 固定协议和停止规则

服务器仅需下列源码与运行资产：

- 用户 fork `https://github.com/Siyuan-Xue/openpi05-bsp.git` 的最终验收 commit；
- 该 commit 锁定的 `third_party/libero` 子模块；
- 官方 `physical-intelligence/libero` LeRobot 数据集 v2.0；
- `pi05_base`、官方 `pi05_libero` checkpoint、PaliGemma tokenizer 和 OpenPI assets。

服务器不需要 BSP 作者仓库、`B-spline.pdf`、`modified_libero_rlds` 或任何原始 RLDS 转换数据。不要在服务器另行 clone/拷贝它们。

下列任一情况出现时立即停止，保留命令、日志和身份信息后报告，不自动改协议：

1. 找不到非系统盘的持久文件系统，或其可用空间小于 500 GiB（推荐至少 1 TiB）。
2. GitHub、Hugging Face、GCS、PyPI、Docker Hub 或 ghcr.io 网络门禁失败。
3. 无法 checkout 已确认的最终 commit，锁文件不一致，或 Python/uv/SciPy 版本不一致。
4. Docker 路线的嵌套 Docker、GPU 透传或 EGL 失败，且隔离双环境备选路线也无法通过 EGL。
5. 数据元数据不是 1,693 episodes / 273,465 frames / 40 tasks / 10 Hz，BSP full verify 失败，或 A/B state norm gate 失败。
6. baseline 或 BSP 在 `micro_batch_size=1` 仍无法运行一个完整 optimizer step。此时不改 LoRA，不降低有效 batch 256。
7. 任一 100-step pilot 失败、正式训练缺少固定里程碑，或评测留有 infrastructure/artifact error。

命令块假定使用 Bash，开始时先启用失败即停和 pipeline 错误传播：

```bash
set -euo pipefail
```

本文不包含 `sudo`、系统 Python/CUDA/驱动安装、`rm`、`docker ... rm`、`docker compose down`、Git reset/clean 或自动覆盖旧实验的命令。所有输出目录必须是新目录；已存在时停止并人工核对，不自动清理。

## 1. 首次登录：只读盘点

本节不安装软件、不拉取镜像、不创建容器。将整段输出保存到本地终端记录。

### 1.1 CPU、内存、H20、驱动与 CUDA 兼容性

```bash
date -u
uname -a
lscpu
free -h
nvidia-smi -L
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version --format=csv,noheader
```

成功判据：GPU 名称明确为 H20，真实显存、驱动版本和 `nvidia-smi` 顶部的“CUDA Version”均有输出。这个 CUDA Version 是驱动支持上限，不要因为它与基础镜像标签不同就安装系统 CUDA。

### 1.2 宿主机/托管容器和磁盘

```bash
cat /proc/1/cgroup
cat /proc/self/mountinfo
df -hT
df -i
lsblk -f
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS
```

根据 `/proc/1/cgroup` 和 mountinfo 判断当前 shell 是否已在托管容器中。根据 `lsblk`/`findmnt` 确认候选数据盘是持久块设备或持久网络文件系统，不是 `/` 所在系统盘、overlay、tmpfs 或会随实例释放的临时盘。

### 1.3 Docker、Compose、NVIDIA Container Toolkit 与 Docker Root Dir

```bash
set +e
docker version
docker compose version
docker info --format 'DockerRootDir={{.DockerRootDir}}'
docker info --format 'Runtimes={{json .Runtimes}} DefaultRuntime={{.DefaultRuntime}}'
nvidia-ctk --version
nvidia-container-cli --version
nvidia-container-cli info
docker image ls --digests
set -e
```

`nvidia-container-cli info` 是本阶段的只读 GPU 容器能力检查。真正的 Docker GPU/EGL 透传在后面的 task-0 冒烟中验收。如果 Docker daemon 无法访问，记录错误并计划使用第 7 节的隔离双环境路线；不修改 daemon 或宿主机配置。

同时核对 Docker Root Dir 所在文件系统能容纳两张镜像和 build cache。如果它位于容量紧张的系统盘，停止并请管理员决定 Docker 数据目录，本 runbook 不自动迁移。

### 1.4 六个外网端点

HTTP `401`/`403` 也能证明 registry 的 DNS/TLS 路径可达；`000` 表示连接未建立。GCS 资产的实际读权限由后面的预取命令再次确认。

```bash
probe_http() {
  local name="$1"
  local url="$2"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 15 --max-time 30 "$url")"
  printf '%-12s %s %s\n' "$name" "$code" "$url"
  test "$code" != 000
}

git ls-remote https://github.com/Siyuan-Xue/openpi05-bsp.git HEAD
probe_http huggingface https://huggingface.co/
probe_http gcs https://storage.googleapis.com/storage/v1/b/openpi-assets
probe_http pypi https://pypi.org/simple/
probe_http dockerhub https://registry-1.docker.io/v2/
probe_http ghcr https://ghcr.io/v2/
```

任一命令失败即停止，不开始下载。

## 2. 选择持久盘并创建目录

只有当第 1 节确认 `/mnt/workspace` 是非系统盘的持久文件系统时，才使用下列推荐值。否则将 `BSP_PARENT` 替换为已核实的持久挂载点，并在创建任何目录前把盘点结果交回确认。

```bash
export BSP_PARENT=/mnt/workspace
test -d "${BSP_PARENT:?}"
findmnt -T "$BSP_PARENT" -o TARGET,SOURCE,FSTYPE,OPTIONS

python3 - "$BSP_PARENT" <<'PY'
import pathlib
import shutil
import sys

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
usage = shutil.disk_usage(path)
minimum = 500 * 1024**3
recommended = 1024**4
print(f"persistent candidate={path}")
print(f"free_GiB={usage.free / 1024**3:.1f}")
if usage.free < minimum:
    raise SystemExit("STOP: persistent filesystem has less than 500 GiB free")
if usage.free < recommended:
    print("NOTICE: gate passes, but 1 TiB free is recommended")
PY
```

手工再次确认 `findmnt` 的 SOURCE/FSTYPE 不是系统盘或临时层，然后创建精确目录：

```bash
export BSP_ROOT=/mnt/workspace/openpi-bsp
export BSP_ROOT="$(realpath -m -- "${BSP_ROOT:?}")"
case "$BSP_ROOT" in
  /|/mnt|/mnt/workspace|/home|/root) echo "STOP: BSP_ROOT is too broad: $BSP_ROOT" >&2; exit 2 ;;
esac
test "$(dirname "$BSP_ROOT")" = "$BSP_PARENT"

mkdir -p \
  "$BSP_ROOT/repo" \
  "$BSP_ROOT/venvs" \
  "$BSP_ROOT/data/bspline-targets" \
  "$BSP_ROOT/cache/huggingface/lerobot/physical-intelligence" \
  "$BSP_ROOT/cache/openpi" \
  "$BSP_ROOT/cache/uv" \
  "$BSP_ROOT/cache/jax" \
  "$BSP_ROOT/experiments/assets" \
  "$BSP_ROOT/experiments/checkpoints" \
  "$BSP_ROOT/experiments/eval" \
  "$BSP_ROOT/experiments/logs" \
  "$BSP_ROOT/experiments/wandb" \
  "$BSP_ROOT/experiments/checkpoints/reserved/seed-43" \
  "$BSP_ROOT/experiments/checkpoints/reserved/seed-44" \
  "$BSP_ROOT/experiments/eval/reserved/seed-43" \
  "$BSP_ROOT/experiments/eval/reserved/seed-44"
```

seed 43/44 目录只作预留，第一阶段不在其中运行任何训练或评测。

目录结构为：

```text
${BSP_ROOT}/
  repo/openpi05-bsp/
  venvs/openpi/
  data/bspline-targets/
  cache/{huggingface,openpi,uv,jax}/
  experiments/{assets,checkpoints,eval,logs,wandb}/
```

官方 LeRobot 数据的精确根目录使用 `${BSP_ROOT}/cache/huggingface/lerobot/physical-intelligence/libero`，BSP sidecar 使用 `${BSP_ROOT}/data/bspline-targets`。观测图像不会复制到 sidecar。

## 3. 锁定源码：只 clone fork 和 LIBERO

在执行前，将 `BSP_CODE_SHA` 替换为交付时明确给出的最终 40 位小写 Git SHA。不要使用浮动分支头进行正式训练。

```bash
export BSP_FORK_URL=https://github.com/Siyuan-Xue/openpi05-bsp.git
export BSP_CODE_SHA=REPLACE_WITH_FINAL_40_HEX_COMMIT
export BSP_REPO_DIR="$BSP_ROOT/repo/openpi05-bsp"

case "$BSP_CODE_SHA" in
  *[!0-9a-f]*|'') echo "STOP: BSP_CODE_SHA must be lowercase hex" >&2; exit 2 ;;
esac
test "${#BSP_CODE_SHA}" -eq 40
test ! -e "$BSP_REPO_DIR"

git clone --filter=blob:none --no-recurse-submodules "$BSP_FORK_URL" "$BSP_REPO_DIR"
git -C "$BSP_REPO_DIR" fetch --depth 1 origin "$BSP_CODE_SHA"
git -C "$BSP_REPO_DIR" checkout --detach "$BSP_CODE_SHA"
test "$(git -C "$BSP_REPO_DIR" rev-parse HEAD)" = "$BSP_CODE_SHA"

git -C "$BSP_REPO_DIR" submodule update --init third_party/libero
git -C "$BSP_REPO_DIR" submodule foreach --quiet 'printf "%s %s\n" "$name" "$(git rev-parse HEAD)"'
test "$(git -C "$BSP_REPO_DIR" submodule foreach --quiet 'printf "%s\n" "$name"')" = third_party/libero

git -C "$BSP_REPO_DIR" status --short --branch
test -z "$(git -C "$BSP_REPO_DIR" status --porcelain --untracked-files=all)"
git -C "$BSP_REPO_DIR" rev-parse HEAD | tee "$BSP_ROOT/experiments/logs/code-sha.txt"
git -C "$BSP_REPO_DIR/third_party/libero" rev-parse HEAD | tee "$BSP_ROOT/experiments/logs/libero-submodule-sha.txt"
```

成功判据：主仓库 HEAD 与 `BSP_CODE_SHA` 完全一致，`submodule foreach` 只输出 `third_party/libero`。`third_party/aloha` 不初始化。

## 4. 独立 OpenPI 3.11.9 环境与持久 cache

### 4.1 每个新 shell 统一导出的变量

```bash
export BSP_ROOT=/mnt/workspace/openpi-bsp
export BSP_REPO_DIR="$BSP_ROOT/repo/openpi05-bsp"
export BSP_CODE_SHA="$(git -C "$BSP_REPO_DIR" rev-parse HEAD)"
export OPENPI_VENV="$BSP_ROOT/venvs/openpi"
export OPENPI_PY="$OPENPI_VENV/bin/python"
export UV_VERSION=0.11.32
export UV_BIN="$BSP_ROOT/venvs/uv-$UV_VERSION-bin/uv"
export LIBERO_DATASET_DIR="$BSP_ROOT/cache/huggingface/lerobot/physical-intelligence/libero"
export BSP_CACHE="$BSP_ROOT/data/bspline-targets/libero-v2.0-bsp-v2.npz"
export BSP_VERIFY="$BSP_ROOT/data/bspline-targets/libero-v2.0-bsp-v2.verification.json"
export ASSETS_BASE="$BSP_ROOT/experiments/assets"
export CHECKPOINT_BASE="$BSP_ROOT/experiments/checkpoints"
export EVAL_BASE="$BSP_ROOT/experiments/eval"
export LOG_BASE="$BSP_ROOT/experiments/logs"

export HF_HOME="$BSP_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export OPENPI_DATA_HOME="$BSP_ROOT/cache/openpi"
export UV_CACHE_DIR="$BSP_ROOT/cache/uv"
export JAX_COMPILATION_CACHE_DIR="$BSP_ROOT/cache/jax"
export WANDB_DIR="$BSP_ROOT/experiments/wandb"
export WANDB_MODE=offline
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```

`OPENPI_DATA_HOME` 同时是宿主机预取和 policy 容器的持久 OpenPI cache。W&B 默认 offline；第一阶段不因外部 W&B 服务状态改变训练协议。

### 4.2 用 uv 0.11.32 创建 Python 3.11.9 环境

不复用基础镜像的 Python 3.12/PyTorch 2.10，不改系统 Python 或 CUDA。uv 仅安装到 `${BSP_ROOT}`：

```bash
cd "$BSP_REPO_DIR"
export UV_VERSION=0.11.32
export UV_BIN_DIR="$BSP_ROOT/venvs/uv-$UV_VERSION-bin"
export UV_INSTALLER="$UV_CACHE_DIR/uv-$UV_VERSION-install.sh"
export UV_PYTHON_INSTALL_DIR="$BSP_ROOT/venvs/python-builds"

test ! -e "$UV_BIN_DIR"
curl -fL "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$UV_INSTALLER"
sha256sum "$UV_INSTALLER" | tee "$LOG_BASE/uv-$UV_VERSION-installer.sha256"
mkdir -p "$UV_BIN_DIR"
env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh "$UV_INSTALLER"
export UV_BIN="$UV_BIN_DIR/uv"
test "$($UV_BIN --version)" = "uv $UV_VERSION"

test "$(cat .python-version)" = 3.11.9
"$UV_BIN" python install 3.11.9
"$UV_BIN" lock --check --offline --python 3.11.9
UV_PROJECT_ENVIRONMENT="$OPENPI_VENV" "$UV_BIN" sync --frozen --python 3.11.9

test "$($OPENPI_PY -c 'import platform; print(platform.python_version())')" = 3.11.9
"$OPENPI_PY" -c 'import scipy; assert scipy.__version__ == "1.15.3"; print(scipy.__version__)'
"$OPENPI_PY" -c 'import jax; print(jax.__version__, jax.devices())'
```

`uv sync` 只在这个安装阶段执行一次。之后所有训练、数据准备、推理和报告命令均直接调用 `${OPENPI_VENV}/bin/python`，避免运行时触发 uv sync 或网络解析。

### 4.3 首次重任务前的 help 和轻量合同门禁

先用实际锁定环境确认 Tyro 参数。训练 config 是位置参数；评测参数因 `eval_libero(args: Args)` 结构而必须使用 `--args.*`；policy checkpoint 使用 union 子命令 `policy:checkpoint`。

```bash
cd "$BSP_REPO_DIR"
mkdir -p "$LOG_BASE/help"

"$OPENPI_PY" scripts/train.py pi05_libero_baseline_h16 --help | tee "$LOG_BASE/help/train-baseline.txt"
"$OPENPI_PY" scripts/train.py pi05_libero_bsp_h16 --help | tee "$LOG_BASE/help/train-bsp.txt"
"$OPENPI_PY" scripts/compute_norm_stats.py --help | tee "$LOG_BASE/help/norm.txt"
"$OPENPI_PY" scripts/prepare_libero_bsp.py --help | tee "$LOG_BASE/help/prepare.txt"
"$OPENPI_PY" scripts/serve_policy.py --help | tee "$LOG_BASE/help/serve.txt"
"$OPENPI_PY" examples/libero/main.py --help | tee "$LOG_BASE/help/evaluator.txt"
PYTHONPATH="$BSP_REPO_DIR/packages/openpi-client/src" \
  "$OPENPI_PY" scripts/compare_libero_phase1.py --help | tee "$LOG_BASE/help/compare.txt"

grep -F -- '--micro-batch-size' "$LOG_BASE/help/train-baseline.txt"
grep -F -- '--assets-base-dir' "$LOG_BASE/help/train-baseline.txt"
grep -F -- '--checkpoint-base-dir' "$LOG_BASE/help/train-baseline.txt"
grep -F -- '--data.lerobot-root' "$LOG_BASE/help/train-baseline.txt"
grep -F -- '--data.bsp-cache-path' "$LOG_BASE/help/train-bsp.txt"
grep -F -- 'policy:checkpoint' "$LOG_BASE/help/serve.txt"
grep -F -- '--args.task-suite-name' "$LOG_BASE/help/evaluator.txt"
grep -F -- '--args.checkpoint-step' "$LOG_BASE/help/evaluator.txt"
```

然后运行不下载数据/模型、不启动仿真的合同和单元测试：

```bash
cd "$BSP_REPO_DIR"
export PYTHONPATH="$BSP_REPO_DIR/src:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR"

"$OPENPI_PY" scripts/server_runtime_contract_test.py
"$OPENPI_PY" scripts/libero_compose_preflight_test.py
"$OPENPI_PY" scripts/libero_revision_contract_test.py
"$OPENPI_PY" scripts/compare_libero_phase1_test.py
"$OPENPI_PY" src/openpi/training/runtime_paths_test.py
"$OPENPI_PY" src/openpi/training/train_planning_test.py

"$OPENPI_PY" -m pytest -q \
  src/openpi/training/bsp_test.py \
  src/openpi/training/bsp_dataset_test.py \
  src/openpi/training/data_loader_test.py \
  src/openpi/policies/libero_policy_test.py \
  scripts/prepare_libero_bsp_test.py \
  scripts/train_test.py \
  scripts/libero_eval_test.py \
  packages/openpi-client/src/openpi_client/libero_eval_test.py \
  packages/openpi-client/src/openpi_client/libero_report_test.py \
  | tee "$LOG_BASE/server-unit-tests.log"
```

成功判据：全部命令返回 0，合同中明确显示 Python 3.11.9、SciPy 1.15.3、LIBERO v2.0、固定 checkpoint 语义和比较器规则。

## 5. 下载官方 LeRobot LIBERO v2.0 和模型资产

### 5.1 数据集

```bash
cd "$BSP_REPO_DIR"
test ! -e "$LIBERO_DATASET_DIR/meta/info.json"

"$OPENPI_PY" scripts/prepare_libero_bsp.py \
  --mode download \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --repo-id physical-intelligence/libero \
  --revision v2.0 \
  2>&1 | tee "$LOG_BASE/libero-v2-download.log"
```

该命令使用官方 LeRobot 数据，不是 `modified_libero_rlds`。它会在返回 0 前校验 1,693 episodes、273,465 frames、40 tasks 和 10 fps。任一数值不一致即停止。

### 5.2 `pi05_base`、官方 `pi05_libero`、tokenizer/OpenPI assets

```bash
cd "$BSP_REPO_DIR"
"$OPENPI_PY" - <<'PY' | tee "$LOG_BASE/openpi-prefetch.log"
from openpi.shared.download import maybe_download

for uri in (
    "gs://openpi-assets/checkpoints/pi05_base",
    "gs://openpi-assets/checkpoints/pi05_libero",
):
    print(uri, "->", maybe_download(uri))
print(
    "gs://big_vision/paligemma_tokenizer.model",
    "->",
    maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}),
)
PY

export BASE_CKPT_LOCAL="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base"
export OFFICIAL_CKPT_LOCAL="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
test -d "$BASE_CKPT_LOCAL/params"
test -d "$OFFICIAL_CKPT_LOCAL/params"

find "$OFFICIAL_CKPT_LOCAL/assets" -type f -name norm_stats.json -print | tee "$LOG_BASE/official-norm-files.txt"
test "$(wc -l < "$LOG_BASE/official-norm-files.txt")" -eq 1
export OFFICIAL_NORM_FILE="$(sed -n '1p' "$LOG_BASE/official-norm-files.txt")"
export OFFICIAL_NORM_HASH="$(sha256sum "$OFFICIAL_NORM_FILE" | awk '{print $1}')"
printf '%s  %s\n' "$OFFICIAL_NORM_HASH" "$OFFICIAL_NORM_FILE" | tee "$LOG_BASE/official-norm.sha256"
```

定义一个只读的目录树摘要函数，记录基座和官方校准 checkpoint 身份：

```bash
tree_sha256() {
  local root="$1"
  test -d "$root"
  (
    cd "$root"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

printf 'pi05_base %s\n' "$(tree_sha256 "$BASE_CKPT_LOCAL")" | tee "$LOG_BASE/pi05-base-tree.sha256"
printf 'pi05_libero %s\n' "$(tree_sha256 "$OFFICIAL_CKPT_LOCAL")" | tee "$LOG_BASE/pi05-libero-tree.sha256"
sha256sum "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" | tee "$LOG_BASE/paligemma-tokenizer.sha256"
```

预取失败就是 GCS/tokenizer 实际读取门禁失败；在训练之前停止。

## 6. 首选路线：Docker + headless EGL

### 6.1 Compose mount preflight 和两张镜像

```bash
cd "$BSP_REPO_DIR"
export BSP_EXPERIMENTS_DIR="$BSP_ROOT/experiments"
export BSP_OPENPI_CACHE_DIR="$BSP_ROOT/cache/openpi"
export BSP_JAX_CACHE_DIR="$BSP_ROOT/cache/jax"
export MUJOCO_GL=egl

"$OPENPI_PY" scripts/libero_compose_preflight.py
docker compose -f examples/libero/compose.yml config --quiet
docker compose -f examples/libero/compose.yml build
```

preflight 会拒绝空值、相对路径、不存在路径、根目录、普通文件和彼此重叠的 bind root。源码在两个容器内都只读；只有 simulator 的 `/experiments` 和 policy server 的 OpenPI/JAX cache 可写。policy 镜像运行时直接执行 `/.venv/bin/python`，不会再调用 uv。

记录两张镜像的真实内容 ID，并按 `openpi_server` 后 `libero` 的固定顺序生成 manifest 要求的单一容器栈摘要：

```bash
export POLICY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' openpi_server)"
export LIBERO_IMAGE_ID="$(docker image inspect --format '{{.Id}}' libero)"
case "$POLICY_IMAGE_ID" in sha256:????????????????????????????????????????????????????????????????) ;; *) exit 2 ;; esac
case "$LIBERO_IMAGE_ID" in sha256:????????????????????????????????????????????????????????????????) ;; *) exit 2 ;; esac

printf 'openpi_server=%s\nlibero=%s\n' "$POLICY_IMAGE_ID" "$LIBERO_IMAGE_ID" \
  | tee "$LOG_BASE/container-images.txt"
export CONTAINER_DIGEST="sha256:$(printf 'openpi_server=%s\nlibero=%s\n' "$POLICY_IMAGE_ID" "$LIBERO_IMAGE_ID" | sha256sum | awk '{print $1}')"
printf 'container_stack=%s\n' "$CONTAINER_DIGEST" | tee "$LOG_BASE/container-stack.sha256"
docker image inspect openpi_server libero > "$LOG_BASE/container-image-inspect.json"
```

`container-images.txt` 保留两个真实 image ID；`CONTAINER_DIGEST` 是两者固定顺序的可复算组合身份，格式严格为 `sha256:` 加 64 位小写十六进制。十次正式评测之间不重建镜像，否则身份门禁将失败。

### 6.2 有界 policy health wait

Compose 不同时启动 server 和 runtime。先启动 policy server，再最多等待 180 秒，反复请求服务实现提供的真实 `/healthz`，只有收到 HTTP 200 和正文 `OK` 才启动 runtime，从而避免 runtime 的 30 秒连接竞态：

```bash
wait_for_policy() {
  "$OPENPI_PY" - 127.0.0.1 8000 180 <<'PY'
import sys
import time
import urllib.error
import urllib.request

host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + timeout
last_error = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2) as response:
            body = response.read().decode("ascii").strip()
            if response.status == 200 and body == "OK":
                print(f"policy /healthz ready at {host}:{port}")
                raise SystemExit(0)
            last_error = RuntimeError(f"unexpected health response: status={response.status}, body={body!r}")
            time.sleep(2)
    except (OSError, urllib.error.URLError) as error:
        last_error = error
        time.sleep(2)
raise SystemExit(f"STOP: policy /healthz was not ready within {timeout}s: {last_error}")
PY
}
```

### 6.3 真正的 task 0 × 1 rollout EGL 冒烟

官方 `pi05_libero` 输出 horizon 10。使用显式 `policy:checkpoint` union 和预取后的容器内路径：

```bash
cd "$BSP_REPO_DIR"
export BSP_CODE_SHA="$(git rev-parse HEAD)"
export OFFICIAL_CONTAINER_CKPT=/openpi_assets/openpi-assets/checkpoints/pi05_libero
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir $OFFICIAL_CONTAINER_CKPT"

docker compose -f examples/libero/compose.yml stop openpi_server
docker compose -f examples/libero/compose.yml up -d openpi_server
wait_for_policy

export OFFICIAL_SMOKE_HOST="$EVAL_BASE/official-h10-task0-smoke"
test ! -e "$OFFICIAL_SMOKE_HOST"
export CLIENT_ARGS="\
--args.host 127.0.0.1 \
--args.port 8000 \
--args.task-suite-name libero_spatial \
--args.task-ids 0 \
--args.policy-variant baseline \
--args.expected-action-horizon 10 \
--args.num-trials-per-task 1 \
--args.output-dir /experiments/eval/official-h10-task0-smoke \
--args.config-name pi05_libero \
--args.checkpoint-step 30000 \
--args.code-sha $BSP_CODE_SHA \
--args.dataset-revision v2.0 \
--args.norm-hash $OFFICIAL_NORM_HASH \
--args.checkpoint $OFFICIAL_CONTAINER_CKPT \
--args.container-digest $CONTAINER_DIGEST \
--args.train-seed 42 \
--args.eval-seed 42"

docker compose -f examples/libero/compose.yml run --no-deps \
  --name libero-official-h10-task0-smoke runtime
docker compose -f examples/libero/compose.yml stop openpi_server
```

注意没有 `--rm`：容器和日志留存供失败审计。同名容器已存在时停止并人工核对，不自动删除。

验证这确实是 1 个 episode，且没有 infrastructure/artifact 未完成：

```bash
"$OPENPI_PY" - "$OFFICIAL_SMOKE_HOST" 1 <<'PY'
import json
import pathlib
import sys

root, expected = pathlib.Path(sys.argv[1]), int(sys.argv[2])
summary = json.loads((root / "summary.json").read_text())
lines = (root / "episodes.jsonl").read_text().splitlines()
assert len(lines) == expected, (len(lines), expected)
assert summary["requested_episodes"] == expected
assert summary["acceptance_complete"] is True
assert summary["incomplete_infrastructure_count"] == 0
assert summary["artifact_error_count"] == 0
artifact_errors = root / "artifact_errors.jsonl"
assert not artifact_errors.exists() or not artifact_errors.read_text().strip()
print("evaluation gate passed", root, expected)
PY
```

该冒烟同时验收 Docker GPU 透传、headless EGL、LIBERO assets、图像旋转/尺寸、7D action、websocket 和日志/视频链路。episode 可以是任务失败，但不得是 infrastructure/artifact 失败。如果 EGL 或嵌套 Docker 失败，保留 `docker compose ... logs openpi_server` 和 runtime 容器日志，然后使用第 7 节；不自动改 GLX，不在宿主机安装图形库。

### 6.4 官方 h10 四套件校准：4 × 10 × 5 = 200

冒烟通过后，使用同一镜像和官方 checkpoint 运行诊断校准：

```bash
cd "$BSP_REPO_DIR"
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir $OFFICIAL_CONTAINER_CKPT"
docker compose -f examples/libero/compose.yml up -d openpi_server
wait_for_policy

export OFFICIAL_CAL_HOST="$EVAL_BASE/official-h10-calibration-all-5"
test ! -e "$OFFICIAL_CAL_HOST"
export CLIENT_ARGS="\
--args.host 127.0.0.1 \
--args.port 8000 \
--args.task-suite-name all \
--args.policy-variant baseline \
--args.expected-action-horizon 10 \
--args.num-trials-per-task 5 \
--args.output-dir /experiments/eval/official-h10-calibration-all-5 \
--args.config-name pi05_libero \
--args.checkpoint-step 30000 \
--args.code-sha $BSP_CODE_SHA \
--args.dataset-revision v2.0 \
--args.norm-hash $OFFICIAL_NORM_HASH \
--args.checkpoint $OFFICIAL_CONTAINER_CKPT \
--args.container-digest $CONTAINER_DIGEST \
--args.train-seed 42 \
--args.eval-seed 42"

docker compose -f examples/libero/compose.yml run --no-deps \
  --name libero-official-h10-calibration-all-5 runtime
docker compose -f examples/libero/compose.yml stop openpi_server

"$OPENPI_PY" - "$OFFICIAL_CAL_HOST" 200 <<'PY'
import json
import pathlib
import sys

root, expected = pathlib.Path(sys.argv[1]), int(sys.argv[2])
summary = json.loads((root / "summary.json").read_text())
assert len((root / "episodes.jsonl").read_text().splitlines()) == expected
assert summary["requested_episodes"] == expected
assert summary["evaluated_suite_count"] == 4
assert summary["all_four_suites_evaluated"] is True
assert summary["acceptance_complete"] is True
assert summary["incomplete_infrastructure_count"] == 0
assert summary["artifact_error_count"] == 0
print("official h10 calibration gate passed", expected)
PY
```

校准成功率只用来发现环境、图像、动作或版本的明显错误，不进入第一阶段 A/B 比较，也不得作为六个 h16/BSP 评测输入。

## 7. 仅在嵌套 Docker 不可用时：隔离双环境备选

本节不与第 6 节混用。OpenPI 训练/服务继续使用 `${OPENPI_VENV}` Python 3.11.9；LIBERO simulator 使用另一个 Python 3.8 环境。不向系统 Python 安装任何包。

先只读检查宿主机 EGL：

```bash
ldconfig -p | grep -E 'libEGL|libGLX|libGL\.so|libglfw|libGLEW'
"$OPENPI_PY" - <<'PY'
import ctypes.util
for name in ("EGL", "GL"):
    value = ctypes.util.find_library(name)
    print(name, value)
    if value is None:
        raise SystemExit(f"STOP: host library {name} is unavailable")
PY
```

缺少 EGL/GL 宿主机库时立即停止并报告，不写入或执行系统包安装命令。存在时创建独立 simulator 环境：

```bash
cd "$BSP_REPO_DIR"
export LIBERO_VENV="$BSP_ROOT/venvs/libero-py38"
export LIBERO_PY="$LIBERO_VENV/bin/python"
test ! -e "$LIBERO_VENV"

"$UV_BIN" python install 3.8
"$UV_BIN" venv --python 3.8 "$LIBERO_VENV"
"$UV_BIN" pip sync \
  --python "$LIBERO_PY" \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  packages/openpi-client/pyproject.toml \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match
"$UV_BIN" pip install --python "$LIBERO_PY" --no-deps -e packages/openpi-client -e third_party/libero

test "$($LIBERO_PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = 3.8
"$UV_BIN" pip freeze --python "$LIBERO_PY" | tee "$LOG_BASE/libero-py38-freeze.txt"

env \
  PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  "$LIBERO_PY" - <<'PY'
import mujoco
context = mujoco.GLContext(16, 16)
context.make_current()
context.free()
print("host EGL context gate passed")
PY
```

备选路线没有 Docker image ID，因此用锁文件、两个 Python 版本、simulator freeze、驱动和 LIBERO 子模块 SHA 生成不变的 host runtime 身份，填入评测 schema 保留的 `container_digest` 字段：

```bash
{
  printf 'mode=isolated-host\n'
  sha256sum uv.lock examples/libero/requirements.txt third_party/libero/requirements.txt packages/openpi-client/pyproject.toml
  "$OPENPI_PY" -c 'import platform; print("openpi_python=" + platform.python_version())'
  "$LIBERO_PY" -c 'import platform; print("libero_python=" + platform.python_version())'
  sha256sum "$LOG_BASE/libero-py38-freeze.txt"
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
  git -C third_party/libero rev-parse HEAD
} | tee "$LOG_BASE/isolated-host-runtime-identity.txt"
export CONTAINER_DIGEST="sha256:$(sha256sum "$LOG_BASE/isolated-host-runtime-identity.txt" | awk '{print $1}')"
```

服务端直接运行：

```bash
"$OPENPI_PY" scripts/serve_policy.py \
  --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "$OFFICIAL_CKPT_LOCAL"
```

在另一个 shell 执行 `wait_for_policy`，然后用与第 6.3/6.4 完全相同的 `--args.*` 协议运行 simulator，只将容器内路径替换为宿主机绝对路径：

```bash
env \
  PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  "$LIBERO_PY" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.task-ids 0 \
  --args.policy-variant baseline \
  --args.expected-action-horizon 10 \
  --args.num-trials-per-task 1 \
  --args.output-dir "$EVAL_BASE/official-h10-task0-smoke-host" \
  --args.config-name pi05_libero \
  --args.checkpoint-step 30000 \
  --args.code-sha "$BSP_CODE_SHA" \
  --args.dataset-revision v2.0 \
  --args.norm-hash "$OFFICIAL_NORM_HASH" \
  --args.checkpoint "$OFFICIAL_CKPT_LOCAL" \
  --args.container-digest "$CONTAINER_DIGEST" \
  --args.train-seed 42 \
  --args.eval-seed 42
```

保持同一个 official policy server 运行，在新的 simulator shell 中执行精确的 200 回合校准命令：

```bash
export OFFICIAL_CAL_HOST="$EVAL_BASE/official-h10-calibration-all-5-host"
test ! -e "$OFFICIAL_CAL_HOST"
env \
  PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  "$LIBERO_PY" examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name all \
  --args.policy-variant baseline \
  --args.expected-action-horizon 10 \
  --args.num-trials-per-task 5 \
  --args.output-dir "$OFFICIAL_CAL_HOST" \
  --args.config-name pi05_libero \
  --args.checkpoint-step 30000 \
  --args.code-sha "$BSP_CODE_SHA" \
  --args.dataset-revision v2.0 \
  --args.norm-hash "$OFFICIAL_NORM_HASH" \
  --args.checkpoint "$OFFICIAL_CKPT_LOCAL" \
  --args.container-digest "$CONTAINER_DIGEST" \
  --args.train-seed 42 \
  --args.eval-seed 42

"$OPENPI_PY" - "$OFFICIAL_CAL_HOST" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
assert len((root / "episodes.jsonl").read_text().splitlines()) == 200
assert summary["requested_episodes"] == 200
assert summary["evaluated_suite_count"] == 4
assert summary["all_four_suites_evaluated"] is True
assert summary["acceptance_complete"] is True
assert summary["incomplete_infrastructure_count"] == 0
assert summary["artifact_error_count"] == 0
print("official host h10 calibration gate passed")
PY
```

必须先完成 task-0 × 1 的真实 host EGL 冒烟，再完成上述 200 回合校准；随后回到 policy server shell 按 `Ctrl-C` 正常停止服务。两个门禁都通过前不进入数据拟合/训练。后面的六次评测同理调用这两个直接 Python 进程，参数与第 12 节一致。

## 8. 构建并全量验证 BSP sidecar

拟合在原始 7D delta-action 上按完整 episode 执行：不跨 episode，不在局部窗口重拟合，不在拟合前归一化。sidecar 仅存控制点/knot 和 timestep mapping，不复制 observation。

代码内固定且写入 fingerprint/评测 manifest 的参数是：cubic degree 3、chunk size 10、target 16 行、7 个 action 维度全部连续拟合（包括 gripper）、controls-first `[16,8]` 通道、严格最大绝对误差 `<0.002`、FITPACK smoothing `1e-12`、stride 1、frame-index 时间轴、`relative_knots=False`、当前 frame 为 materialized knot 原点、projection epsilon `1e-6`、模型 action dim 32/horizon 16、只用前 12 个 control points 并在 `[knots[3], knots[-4]]` 内无外推地解码/执行 8 步。训练不增加 reconstruction、smoothness 或 monotonicity loss。

```bash
cd "$BSP_REPO_DIR"
test ! -e "$BSP_CACHE"
test ! -e "$BSP_VERIFY"

"$OPENPI_PY" scripts/prepare_libero_bsp.py \
  --mode build \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --cache-path "$BSP_CACHE" \
  --repo-id physical-intelligence/libero \
  --revision v2.0 \
  --action-key actions \
  2>&1 | tee "$LOG_BASE/bsp-build.log"

"$OPENPI_PY" scripts/prepare_libero_bsp.py \
  --mode verify \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --cache-path "$BSP_CACHE" \
  --diagnostics-path "$BSP_VERIFY" \
  --repo-id physical-intelligence/libero \
  --revision v2.0 \
  --action-key actions \
  2>&1 | tee "$LOG_BASE/bsp-verify.log"
```

`verify` 会全量重建每个 episode 并校验：严格 `max_abs_error < 0.002`、ground-truth knots 非递减、tail padding、最近未来 mapping、target bounds、不跨 episode、全 frame 覆盖、targets/mapping 与重建一致和 cache 内容确定性。诊断 JSON 还包含真实 NPZ SHA256、cache manifest fingerprint、SciPy 1.15.3、误差统计和 code SHA。

用原始 verify JSON 生成后续需要的 shell 身份，不复制、不编辑这个 JSON：

```bash
export BSP_CACHE_HASH="$(sha256sum "$BSP_CACHE" | awk '{print $1}')"
export BSP_CACHE_MANIFEST_FINGERPRINT="$(
  "$OPENPI_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_manifest_fingerprint"])' "$BSP_VERIFY"
)"

"$OPENPI_PY" - "$BSP_VERIFY" "$BSP_CACHE_HASH" "$BSP_CACHE_MANIFEST_FINGERPRINT" <<'PY'
import json
import re
import sys

path, cache_hash, fingerprint = sys.argv[1:]
payload = json.load(open(path))
assert payload["verification_passed"] is True
assert payload["scipy_version"] == "1.15.3"
assert payload["cache_sha256"] == cache_hash
assert payload["cache_manifest_fingerprint"] == fingerprint
assert re.fullmatch(r"[0-9a-f]{64}", cache_hash)
assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
assert payload["strict_max_reconstruction_error"] < payload["max_error_threshold"] == 0.002
print("BSP full verification gate passed")
PY
```

`$BSP_VERIFY` 是第 13 节比较器必须直接读取的 Task-6 原始诊断产物。

## 9. 分别计算 A/B norm stats 并通过 state gate

baseline 的 action stats 来自原始 `[16,7]` action，BSP 的 action stats 来自归一化前的 `[16,8]` controls-first target。两者使用独立 asset ID，但 observation/state 必须完全一致。

```bash
cd "$BSP_REPO_DIR"
export BASELINE_CONFIG_ASSETS="$ASSETS_BASE/pi05_libero_baseline_h16"
export BSP_CONFIG_ASSETS="$ASSETS_BASE/pi05_libero_bsp_h16"
export BASELINE_NORM_DIR="$BASELINE_CONFIG_ASSETS/libero_baseline_h16"
export BSP_NORM_DIR="$BSP_CONFIG_ASSETS/libero_bsp_h16"
export NORM_COMPARISON="$ASSETS_BASE/libero-phase1-norm-comparison.json"

test ! -e "$BASELINE_NORM_DIR/norm_stats.json"
test ! -e "$BSP_NORM_DIR/norm_stats.json"
test ! -e "$NORM_COMPARISON"

"$OPENPI_PY" scripts/compute_norm_stats.py \
  --config-name pi05_libero_baseline_h16 \
  --assets-dir "$BASELINE_CONFIG_ASSETS" \
  --dataset-root "$LIBERO_DATASET_DIR" \
  2>&1 | tee "$LOG_BASE/norm-baseline.log"

"$OPENPI_PY" scripts/compute_norm_stats.py \
  --config-name pi05_libero_bsp_h16 \
  --assets-dir "$BSP_CONFIG_ASSETS" \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --bsp-cache-path "$BSP_CACHE" \
  --compare-state-stats-with "$BASELINE_NORM_DIR" \
  --norm-comparison-output "$NORM_COMPARISON" \
  2>&1 | tee "$LOG_BASE/norm-bsp.log"
```

对原始 comparison JSON 执行门禁：

```bash
"$OPENPI_PY" - "$NORM_COMPARISON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
assert payload["state_stats_equal"] is True
assert payload["asset_directories_isolated"] is True
assert payload["action_stats_isolated"] is True
for field in ("mean", "std", "q01", "q99"):
    assert payload["state_fields"][field]["equal"] is True
print("A/B normalization gate passed")
PY

export BASELINE_NORM_HASH="$(sha256sum "$BASELINE_NORM_DIR/norm_stats.json" | awk '{print $1}')"
export BSP_NORM_HASH="$(sha256sum "$BSP_NORM_DIR/norm_stats.json" | awk '{print $1}')"
printf 'baseline %s\nbsp %s\n' "$BASELINE_NORM_HASH" "$BSP_NORM_HASH" \
  | tee "$LOG_BASE/phase1-norm.sha256"
```

`$NORM_COMPARISON` 是第 13 节比较器必须直接读取的另一个原始诊断产物。不要手工复制或修改其字段。

## 10. H20 显存探针：`{1,2,4,8}` 独立进程

探针前停止 policy server，确认 H20 没有遗留训练/推理进程。每个候选值、每个 variant 都使用一个新 Python 进程和唯一 checkpoint 目录。`batch_size=256` 始终是全局有效 batch；`micro_batch_size` 只改变累积次数。

```bash
cd "$BSP_REPO_DIR"
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  docker compose -f examples/libero/compose.yml stop openpi_server
fi
nvidia-smi

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PROBE_CHECKPOINT_BASE="$CHECKPOINT_BASE/probes"
mkdir -p "$PROBE_CHECKPOINT_BASE"

run_probe() {
  local variant="$1"
  local micro="$2"
  local config
  local extra=()
  if test "$variant" = baseline; then
    config=pi05_libero_baseline_h16
  else
    config=pi05_libero_bsp_h16
    extra=(--data.bsp-cache-path "$BSP_CACHE")
  fi
  local exp="probe-${variant}-mb${micro}"
  local output="$PROBE_CHECKPOINT_BASE/$config/$exp"
  if test -e "$output"; then
    echo "STOP: probe output already exists: $output" >&2
    return 2
  fi

  nvidia-smi \
    --query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv -l 1 > "$LOG_BASE/${exp}-nvidia-smi.csv" &
  local monitor_pid=$!
  set +e
  "$OPENPI_PY" scripts/train.py "$config" \
    --exp-name "$exp" \
    --seed 42 \
    --batch-size 256 \
    --micro-batch-size "$micro" \
    --num-train-steps 1 \
    --assets-base-dir "$ASSETS_BASE" \
    --checkpoint-base-dir "$PROBE_CHECKPOINT_BASE" \
    --data.lerobot-root "$LIBERO_DATASET_DIR" \
    "${extra[@]}" \
    2>&1 | tee "$LOG_BASE/${exp}.log"
  local train_rc=${PIPESTATUS[0]}
  kill "$monitor_pid"
  wait "$monitor_pid" 2>/dev/null
  set -e
  if test "$train_rc" -ne 0; then
    return "$train_rc"
  fi
  if ! test -d "$output/1"; then
    echo "STOP: successful probe did not publish checkpoint 1: $output" >&2
    return 3
  fi
  return 0
}

test ! -e "$LOG_BASE/micro-batch-probe-results.tsv"
: > "$LOG_BASE/micro-batch-probe-results.tsv"
for micro in 1 2 4 8; do
  if run_probe baseline "$micro"; then baseline_rc=0; else baseline_rc=$?; fi
  if run_probe bsp "$micro"; then bsp_rc=0; else bsp_rc=$?; fi
  printf '%s\t%s\t%s\n' "$micro" "$baseline_rc" "$bsp_rc" \
    | tee -a "$LOG_BASE/micro-batch-probe-results.tsv"
  if test "$micro" -eq 1 && { test "$baseline_rc" -ne 0 || test "$bsp_rc" -ne 0; }; then
    echo "STOP: micro-batch 1 failed for baseline or BSP" >&2
    exit 2
  fi
done
```

条件分支只在 Docker/Compose 命令存在时停止 policy service；它不吞掉 stop 失败。除 micro-batch 1 是立即停止门禁外，2/4/8 的非零状态会被完整记录，以便确实完成四个候选的独立探测；任何失败都不会被误判为稳定。探针输出目录不会进入正式比较。

选择规则：

1. 两个 variant 都必须返回 0、完成 checkpoint `1`，日志无 OOM/NaN/Inf。
2. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` 给非 JAX/EGL/波动留出上限外的余量；同时核对采样 CSV 没有外部 GPU 进程竞争。
3. 从 8、4、2、1 中选择 A/B 共同稳定的最大值。不允许两组使用不同值。
4. 如果任一 variant 在 1 失败，立即停止并报告硬件阻塞；不自动改 LoRA、action horizon、action dim 或有效 batch 256。

手工选定后记录：

```bash
export MICRO_BATCH=REPLACE_WITH_1_2_4_OR_8
case "$MICRO_BATCH" in 1|2|4|8) ;; *) echo "STOP: invalid MICRO_BATCH" >&2; exit 2 ;; esac
test "$(awk '$2 == 0 && $3 == 0 {largest=$1} END {print largest}' "$LOG_BASE/micro-batch-probe-results.tsv")" = "$MICRO_BATCH"
printf 'effective_batch=256\nmicro_batch=%s\nseed=42\n' "$MICRO_BATCH" \
  | tee "$LOG_BASE/selected-micro-batch.txt"
```

## 11. 两个 100 optimizer-step pilot

pilot 也是两个独立进程，它们使用正式 norm、同一 `pi05_base`、seed 42、effective batch 256 和已选的共同 micro-batch，但 checkpoint 与正式训练隔离。

```bash
cd "$BSP_REPO_DIR"
export PILOT_CHECKPOINT_BASE="$CHECKPOINT_BASE/pilots"
mkdir -p "$PILOT_CHECKPOINT_BASE"

test ! -e "$PILOT_CHECKPOINT_BASE/pi05_libero_baseline_h16/pilot-seed42-baseline"
"$OPENPI_PY" scripts/train.py pi05_libero_baseline_h16 \
  --exp-name pilot-seed42-baseline \
  --seed 42 \
  --batch-size 256 \
  --micro-batch-size "$MICRO_BATCH" \
  --num-train-steps 100 \
  --assets-base-dir "$ASSETS_BASE" \
  --checkpoint-base-dir "$PILOT_CHECKPOINT_BASE" \
  --data.lerobot-root "$LIBERO_DATASET_DIR" \
  2>&1 | tee "$LOG_BASE/pilot-seed42-baseline.log"

test ! -e "$PILOT_CHECKPOINT_BASE/pi05_libero_bsp_h16/pilot-seed42-bsp"
"$OPENPI_PY" scripts/train.py pi05_libero_bsp_h16 \
  --exp-name pilot-seed42-bsp \
  --seed 42 \
  --batch-size 256 \
  --micro-batch-size "$MICRO_BATCH" \
  --num-train-steps 100 \
  --assets-base-dir "$ASSETS_BASE" \
  --checkpoint-base-dir "$PILOT_CHECKPOINT_BASE" \
  --data.lerobot-root "$LIBERO_DATASET_DIR" \
  --data.bsp-cache-path "$BSP_CACHE" \
  2>&1 | tee "$LOG_BASE/pilot-seed42-bsp.log"

test -d "$PILOT_CHECKPOINT_BASE/pi05_libero_baseline_h16/pilot-seed42-baseline/100"
test -d "$PILOT_CHECKPOINT_BASE/pi05_libero_bsp_h16/pilot-seed42-bsp/100"
```

成功判据：两个进程均返回 0，步数是 optimizer step 而不是 micro-step，各自生成准确标号的 `100` checkpoint，loss/grad norm 为有限值，且每个有效 batch 只更新一次 optimizer/EMA/state step。任一 pilot 失败就不开始正式 10k。

## 12. 两次正式 10k 训练和十个固定评测

### 12.0 长任务重连后的身份恢复

正式阶段通常跨越多次 SSH 会话。每次重连先重新执行第 4.1 节的基础变量块，再从原始产物重新派生下列值；不要从聊天记录手工抄 hash：

```bash
cd "$BSP_REPO_DIR"
export BSP_CODE_SHA="$(git rev-parse HEAD)"
test "$BSP_CODE_SHA" = "$(cat "$LOG_BASE/code-sha.txt")"
export BSP_CACHE_HASH="$(sha256sum "$BSP_CACHE" | awk '{print $1}')"
export BSP_CACHE_MANIFEST_FINGERPRINT="$(
  "$OPENPI_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_manifest_fingerprint"])' "$BSP_VERIFY"
)"
export TRAINING_FAMILY=lora
case "$TRAINING_FAMILY" in
  full)
    export BASELINE_CONFIG=pi05_libero_baseline_h16
    export BSP_CONFIG=pi05_libero_bsp_h16
    ;;
  lora)
    export BASELINE_CONFIG=pi05_libero_baseline_lora_h16
    export BSP_CONFIG=pi05_libero_bsp_lora_h16
    ;;
  *) echo "STOP: TRAINING_FAMILY must be full or lora" >&2; exit 2 ;;
esac
export BASELINE_NORM_DIR="$ASSETS_BASE/$BASELINE_CONFIG/libero_baseline_h16"
export BSP_NORM_DIR="$ASSETS_BASE/$BSP_CONFIG/libero_bsp_h16"
export NORM_COMPARISON="$ASSETS_BASE/libero-phase1-norm-comparison.json"
export BASELINE_NORM_HASH="$(sha256sum "$BASELINE_NORM_DIR/norm_stats.json" | awk '{print $1}')"
export BSP_NORM_HASH="$(sha256sum "$BSP_NORM_DIR/norm_stats.json" | awk '{print $1}')"
export MICRO_BATCH="$(awk -F= '$1 == "micro_batch" {print $2}' "$LOG_BASE/selected-micro-batch.txt")"
case "$MICRO_BATCH" in 1|2|4|8|16|32|64) ;; *) echo "STOP: invalid recovered MICRO_BATCH" >&2; exit 2 ;; esac
export BASELINE_EXP=phase1-short10k-seed42-baseline
export BSP_EXP=phase1-short10k-seed42-bsp
export BASELINE_RUN="$CHECKPOINT_BASE/$BASELINE_CONFIG/$BASELINE_EXP"
export BSP_RUN="$CHECKPOINT_BASE/$BSP_CONFIG/$BSP_EXP"

"$OPENPI_PY" - "$BSP_CODE_SHA" "$BSP_CACHE_HASH" "$BSP_CACHE_MANIFEST_FINGERPRINT" "$BASELINE_NORM_HASH" "$BSP_NORM_HASH" <<'PY'
import re
import sys

code_sha, *hashes = sys.argv[1:]
assert re.fullmatch(r"[0-9a-f]{40}", code_sha)
assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
print("phase-one identities recovered")
PY
```

Docker 路线还要从当前两张镜像重新计算第 6.1 节的组合身份，并确认与首次记录相同：

```bash
export POLICY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' openpi_server)"
export LIBERO_IMAGE_ID="$(docker image inspect --format '{{.Id}}' libero)"
export CONTAINER_DIGEST="sha256:$(printf 'openpi_server=%s\nlibero=%s\n' "$POLICY_IMAGE_ID" "$LIBERO_IMAGE_ID" | sha256sum | awk '{print $1}')"
test "container_stack=$CONTAINER_DIGEST" = "$(cat "$LOG_BASE/container-stack.sha256")"
```

隔离双环境路线则重新执行第 7 节的 host runtime 身份生成块并确认摘要未变化。两条路线二选一，整个十-run 评测过程中不得切换。

### 12.1 正式训练

一张 H20 上顺序启动两个独立进程，不并行。如服务器已有 `tmux`，可在其中执行；没有时使用能保持会话的阿里云终端，不为此安装系统包。

```bash
cd "$BSP_REPO_DIR"
test "$(cat "$LOG_BASE/code-sha.txt")" = "$(git rev-parse HEAD)"
test "$(git rev-parse HEAD)" = "$BSP_CODE_SHA"
git diff --quiet
git diff --cached --quiet
nvidia-smi

export BASELINE_EXP=phase1-short10k-seed42-baseline
export BSP_EXP=phase1-short10k-seed42-bsp
export BASELINE_RUN="$CHECKPOINT_BASE/$BASELINE_CONFIG/$BASELINE_EXP"
export BSP_RUN="$CHECKPOINT_BASE/$BSP_CONFIG/$BSP_EXP"
test ! -e "$BASELINE_RUN"
test ! -e "$BSP_RUN"

"$OPENPI_PY" - "$BASELINE_CONFIG" "$BSP_CONFIG" <<'PY'
import sys
from openpi.training import config as train_config

expected = (0, 1_000, 2_000, 5_000, 10_000)
for name in sys.argv[1:]:
    config = train_config.get_config(name)
    assert config.permanent_checkpoint_steps == expected
    assert config.keep_period == 10_000
print("permanent_checkpoint_steps=0k/1k/2k/5k/10k")
PY

"$OPENPI_PY" scripts/train.py "$BASELINE_CONFIG" \
  --exp-name "$BASELINE_EXP" \
  --seed 42 \
  --batch-size 256 \
  --micro-batch-size "$MICRO_BATCH" \
  --num-train-steps 10000 \
  --save-interval 1000 \
  --ema-decay None \
  --assets-base-dir "$ASSETS_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_BASE" \
  --data.lerobot-root "$LIBERO_DATASET_DIR" \
  2>&1 | tee "$LOG_BASE/train-phase1-short10k-seed42-baseline.log"

"$OPENPI_PY" scripts/train.py "$BSP_CONFIG" \
  --exp-name "$BSP_EXP" \
  --seed 42 \
  --batch-size 256 \
  --micro-batch-size "$MICRO_BATCH" \
  --num-train-steps 10000 \
  --save-interval 1000 \
  --ema-decay None \
  --assets-base-dir "$ASSETS_BASE" \
  --checkpoint-base-dir "$CHECKPOINT_BASE" \
  --data.lerobot-root "$LIBERO_DATASET_DIR" \
  --data.bsp-cache-path "$BSP_CACHE" \
  2>&1 | tee "$LOG_BASE/train-phase1-short10k-seed42-bsp.log"
```

命令不使用 `--overwrite`。如服务器中断，先检查已保存目录是完成 optimizer-step 边界，再在原命令末尾加 `--resume`；不删除、改名或覆盖原运行。每 1,000 optimizer steps 保存当时最新恢复点。配置中的 `permanent_checkpoint_steps=(0, 1000, 2000, 5000, 10000)` 与 `keep_period=10000` 取并集，精确保留 0k/1k/2k/5k/10k。

正式进程启动后，先在另一个 shell 验收尚未训练的 step 0；两个 variant 都必须在第一次正梯度更新前产生同结构的 `0/params` 与 `0/train_state`。step 0 使用各自 norm/推理协议，不能用官方 `pi05_libero` 代替：

```bash
for root in "$BASELINE_RUN" "$BSP_RUN"; do
  test -d "$root/0/params"
  test -d "$root/0/train_state"
  test "$(basename "$root/0")" = 0
done
echo "formal_step_zero_gate=PASS"
```

训练完成门禁：

```bash
for root in "$BASELINE_RUN" "$BSP_RUN"; do
  for step in 0 1000 2000 5000 10000; do
    test -d "$root/$step/params"
    test -d "$root/$step/assets"
  done
done

for step in 0 1000 2000 5000 10000; do
  test "$(sha256sum "$BASELINE_RUN/$step/assets/libero_baseline_h16/norm_stats.json" | awk '{print $1}')" = "$BASELINE_NORM_HASH"
  test "$(sha256sum "$BSP_RUN/$step/assets/libero_bsp_h16/norm_stats.json" | awk '{print $1}')" = "$BSP_NORM_HASH"
done

for variant in baseline bsp; do
  if test "$variant" = baseline; then root="$BASELINE_RUN"; else root="$BSP_RUN"; fi
  for step in 0 1000 2000 5000 10000; do
    printf '%s %s %s\n' "$variant" "$step" "$(tree_sha256 "$root/$step")"
  done
done | tee "$LOG_BASE/phase1-checkpoints.sha256"
```

十个 checkpoint 的路径不得是 symlink 到同一目录，末级目录名必须分别为 `0`、`1000`、`2000`、`5000`、`10000`。

### 12.2 Docker 路线的十次正式评测

每次评测是 4 suites × 10 tasks × 50 initial states = 2,000 episodes，共 20,000 episodes。baseline server 输出严格 horizon 16，执行前 8 步；BSP server 在反归一化后解码为严格 horizon 8。两者均在 10 Hz 下每 8 步重规划。

定义通用完整性检查：

```bash
validate_eval_run() {
  "$OPENPI_PY" - "$1" "$2" <<'PY'
import json
import pathlib
import sys

root, expected = pathlib.Path(sys.argv[1]), int(sys.argv[2])
manifest = json.loads((root / "manifest.json").read_text())
summary = json.loads((root / "summary.json").read_text())
records = (root / "episodes.jsonl").read_text().splitlines()
assert manifest["schema_version"] == 2
assert len(records) == expected
assert summary["requested_episodes"] == expected
assert summary["eligible_episodes"] == expected
assert summary["evaluated_suite_count"] == 4
assert summary["all_four_suites_evaluated"] is True
assert summary["acceptance_complete"] is True
assert summary["incomplete_infrastructure_count"] == 0
assert summary["artifact_error_count"] == 0
artifact_errors = root / "artifact_errors.jsonl"
assert not artifact_errors.exists() or not artifact_errors.read_text().strip()
print("full evaluation gate passed", root, expected)
PY
}
```

定义一次严格评测。注意 checkpoint 同时有宿主机路径和容器内 `/experiments/...` 路径；manifest 必须记录后者，且末级正好是 step：

```bash
run_phase1_eval() {
  local variant="$1"
  local step="$2"
  local config host_root container_root asset_id expected_horizon norm_hash cache_args

  case "$variant" in
    baseline)
      config="$BASELINE_CONFIG"
      host_root="$BASELINE_RUN"
      container_root="/experiments/checkpoints/$BASELINE_CONFIG/$BASELINE_EXP"
      asset_id=libero_baseline_h16
      expected_horizon=16
      cache_args=""
      ;;
    bsp)
      config="$BSP_CONFIG"
      host_root="$BSP_RUN"
      container_root="/experiments/checkpoints/$BSP_CONFIG/$BSP_EXP"
      asset_id=libero_bsp_h16
      expected_horizon=8
      cache_args="--args.bsp-cache-hash $BSP_CACHE_HASH --args.bsp-cache-manifest-fingerprint $BSP_CACHE_MANIFEST_FINGERPRINT"
      ;;
    *) echo "STOP: unsupported variant $variant" >&2; return 2 ;;
  esac
  case "$step" in 0|1000|2000|5000|10000) ;; *) echo "STOP: unsupported step $step" >&2; return 2 ;; esac

  local host_checkpoint="$host_root/$step"
  local container_checkpoint="$container_root/$step"
  local output_name="${variant}-step-${step}"
  local host_output="$EVAL_BASE/$output_name"
  test -d "$host_checkpoint"
  test "$(basename "$host_checkpoint")" = "$step"
  test ! -e "$host_output"

  norm_hash="$(sha256sum "$host_checkpoint/assets/$asset_id/norm_stats.json" | awk '{print $1}')"
  export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config $config --policy.dir $container_checkpoint"
  docker compose -f examples/libero/compose.yml stop openpi_server
  docker compose -f examples/libero/compose.yml up -d openpi_server
  wait_for_policy

  export CLIENT_ARGS="\
--args.host 127.0.0.1 \
--args.port 8000 \
--args.task-suite-name all \
--args.policy-variant $variant \
--args.expected-action-horizon $expected_horizon \
--args.num-trials-per-task 50 \
--args.output-dir /experiments/eval/$output_name \
--args.config-name $config \
--args.checkpoint-step $step \
--args.code-sha $BSP_CODE_SHA \
--args.dataset-revision v2.0 \
--args.norm-hash $norm_hash \
--args.checkpoint $container_checkpoint \
--args.container-digest $CONTAINER_DIGEST \
--args.train-seed 42 \
--args.eval-seed 42 \
$cache_args"

  set +e
  docker compose -f examples/libero/compose.yml run --no-deps \
    --name "libero-${variant}-step-${step}" runtime
  local eval_rc=$?
  docker compose -f examples/libero/compose.yml stop openpi_server
  set -e
  test "$eval_rc" -eq 0
  validate_eval_run "$host_output" 2000
}
```

按固定里程碑执行，不根据中间结果更改后续顺序或跳过 checkpoint：

```bash
cd "$BSP_REPO_DIR"
run_phase1_eval baseline 0
run_phase1_eval bsp 0
run_phase1_eval baseline 1000
run_phase1_eval bsp 1000
run_phase1_eval baseline 2000
run_phase1_eval bsp 2000
run_phase1_eval baseline 5000
run_phase1_eval bsp 5000
run_phase1_eval baseline 10000
run_phase1_eval bsp 10000
```

评测器对 simulator/container/network 错误使用原种子最多重试两次。两次后仍失败的 episode 不混入成功率，且整个 run 的 `acceptance_complete=false`，本节随即停止。非法 shape/NaN/无效 BSP 解码是策略失败，正常计入失败 episode，不伪装为基础设施错误。

每个 episode 先追加到 JSONL；每个 task 只保留首个成功和首个失败视频，路径包含 suite/task/init-state 身份且不得碰撞。被选中的视频编码失败会写入 `artifact_errors.jsonl` 并使整个 run 验收未完成，不能忽略。

每个 run 的 manifest 必须满足：

- `code_sha` 是当前最终 checkout 的 40 位小写 SHA；
- `dataset_revision` 精确为 `v2.0`；
- `container_digest` 是 Docker 路线第 6.1 节的容器栈摘要，或双环境路线第 7 节的 host runtime 摘要；两者均为可复算的 `sha256:<64hex>`；
- checkpoint 路径十个全部唯一，且末级等于 `checkpoint_step`；
- baseline 的两个 BSP cache 身份为 `null`，BSP 的 NPZ hash/fingerprint 来自第 8 节原始产物；
- norm hash 来自当前 checkpoint 内实际 `norm_stats.json`；
- A/B 均使用 train/eval seed 42 和相同初始状态/确定性 flow noise 派生规则。

### 12.3 隔离双环境路线的十次正式评测

只在第 7 节路线已经完成 task-0 冒烟和 200 回合校准时使用本节。它与第 12.2 节二选一，输出目录名保持完全相同，因此无法意外把两条 runtime 路线混入同一报告。

```bash
run_phase1_eval_host() {
  local variant="$1"
  local step="$2"
  local config host_root asset_id expected_horizon norm_hash
  local cache_args=()

  case "$variant" in
    baseline)
      config="$BASELINE_CONFIG"
      host_root="$BASELINE_RUN"
      asset_id=libero_baseline_h16
      expected_horizon=16
      ;;
    bsp)
      config="$BSP_CONFIG"
      host_root="$BSP_RUN"
      asset_id=libero_bsp_h16
      expected_horizon=8
      cache_args=(
        --args.bsp-cache-hash "$BSP_CACHE_HASH"
        --args.bsp-cache-manifest-fingerprint "$BSP_CACHE_MANIFEST_FINGERPRINT"
      )
      ;;
    *) echo "STOP: unsupported variant $variant" >&2; return 2 ;;
  esac
  case "$step" in 0|1000|2000|5000|10000) ;; *) echo "STOP: unsupported step $step" >&2; return 2 ;; esac

  local checkpoint="$host_root/$step"
  local output="$EVAL_BASE/${variant}-step-${step}"
  local server_log="$LOG_BASE/serve-${variant}-step-${step}-host.log"
  test -d "$checkpoint"
  test "$(basename "$checkpoint")" = "$step"
  test ! -e "$output"
  test ! -e "$server_log"
  norm_hash="$(sha256sum "$checkpoint/assets/$asset_id/norm_stats.json" | awk '{print $1}')"

  "$OPENPI_PY" scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config "$config" \
    --policy.dir "$checkpoint" \
    > "$server_log" 2>&1 &
  local server_pid=$!

  set +e
  wait_for_policy
  local ready_rc=$?
  set -e
  if test "$ready_rc" -ne 0; then
    set +e
    kill "$server_pid"
    wait "$server_pid" 2>/dev/null
    set -e
    return "$ready_rc"
  fi
  kill -0 "$server_pid"

  set +e
  env \
    PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    "$LIBERO_PY" examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port 8000 \
    --args.task-suite-name all \
    --args.policy-variant "$variant" \
    --args.expected-action-horizon "$expected_horizon" \
    --args.num-trials-per-task 50 \
    --args.output-dir "$output" \
    --args.config-name "$config" \
    --args.checkpoint-step "$step" \
    --args.code-sha "$BSP_CODE_SHA" \
    --args.dataset-revision v2.0 \
    --args.norm-hash "$norm_hash" \
    --args.checkpoint "$checkpoint" \
    --args.container-digest "$CONTAINER_DIGEST" \
    --args.train-seed 42 \
    --args.eval-seed 42 \
    "${cache_args[@]}"
  local eval_rc=$?
  kill "$server_pid"
  wait "$server_pid" 2>/dev/null
  set -e
  test "$eval_rc" -eq 0
  validate_eval_run "$output" 2000
}
```

`wait_for_policy` 仍然先做最长 180 秒的有界等待；client 完成后只终止本函数刚启动的 policy PID，不删除文件或环境。按固定顺序执行：

```bash
cd "$BSP_REPO_DIR"
run_phase1_eval_host baseline 0
run_phase1_eval_host bsp 0
run_phase1_eval_host baseline 1000
run_phase1_eval_host bsp 1000
run_phase1_eval_host baseline 2000
run_phase1_eval_host bsp 2000
run_phase1_eval_host baseline 5000
run_phase1_eval_host bsp 5000
run_phase1_eval_host baseline 10000
run_phase1_eval_host bsp 10000
```

## 13. 严格生成第一阶段比较报告

比较器只接受恰好十个 run，并且只根据 manifest 的 variant/step 识别它们。它会拒绝 official h10、缺失 0k/1k/2k/5k/10k、重复/额外里程碑、混合全量/LoRA 训练家族、不完整 20,000 回合、非配对初始状态、截断/NaN JSON、summary 不一致、infrastructure/artifact error 或任何身份不一致。

直接传入第 8 节 prepare verify 生成的原始 `$BSP_VERIFY` 和第 9 节生成的原始 `$NORM_COMPARISON`：

```bash
cd "$BSP_REPO_DIR"
export REPORT_DIR="$EVAL_BASE/phase1-comparison-seed42"
test ! -e "$REPORT_DIR"

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

find "$REPORT_DIR" -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort
sha256sum \
  "$REPORT_DIR/task_comparison.csv" \
  "$REPORT_DIR/suite_comparison.csv" \
  "$REPORT_DIR/learning_curve.csv" \
  "$REPORT_DIR/comparison.json" \
  "$REPORT_DIR/report.md" \
  "$REPORT_DIR/learning_curve.svg" \
  | tee "$LOG_BASE/phase1-report.sha256"
```

成功时只会产生六个文件：

```text
task_comparison.csv
suite_comparison.csv
learning_curve.csv
comparison.json
report.md
learning_curve.svg
```

报告固定显示 0k/1k/2k/5k/10k 五个点，不标记 best checkpoint。主指标是 task、suite 和四套件分层宏平均成功率；BSP-baseline 差值使用 seed 42、恰好 10,000 次的 task-stratified paired bootstrap 生成双侧 95% percentile CI。完成步数、推理延迟和 spline 重建误差只作诊断。

## 14. 最终审计清单

只有下列全部为真时，第一阶段才完成：

- 持久盘、网络、H20/驱动、容器或双环境、EGL 门禁通过。
- 最终 code SHA、LIBERO 子模块 SHA、uv 0.11.32、Python 3.11.9、SciPy 1.15.3、`uv.lock` 均有记录。
- 官方 LeRobot LIBERO v2.0 是 1,693 episodes / 273,465 frames / 40 tasks / 10 Hz，不存在 RLDS 替代数据。
- task-0 × 1 EGL 冒烟和 official h10 200 回合校准均无 infrastructure/artifact error。
- BSP full verify 原始 JSON 通过，NPZ hash、manifest fingerprint、严格误差和所有 mapping/cache 不变量均有记录。
- baseline/BSP norm 分离，state `mean/std/q01/q99` 一致，action stats 隔离。
- `{1,2,4,8}` 在 A/B 中全部以独立进程探测，选择共同稳定且有余量的最大值；有效 batch 一直为 256。
- A/B 各 100 optimizer-step pilot 通过，随后从同一 `pi05_base` 用 seed 42 分别完成 10k。
- 十个不同 checkpoint 的 0k/1k/2k/5k/10k 评测各 2,000 回合，总计 20,000，且全部可审计/可配对。
- 比较器直接读取原始 BSP/norm diagnostics，返回 0 并且只生成六个固定报告产物。
- seed 43/44 只有预留目录，没有实际运行；没有评测 2×/4×，没有选择 best checkpoint。

保留 `${BSP_ROOT}/experiments/logs`、十个评测目录、两个原始 diagnostics、十个 checkpoint 和六个报告文件。它们共同构成第一阶段验收证据，不要在报告生成后重写其中任何身份文件。
