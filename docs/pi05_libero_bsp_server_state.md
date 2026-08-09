# π0.5 + LIBERO BSP 服务器实际状态与使用手册

> 状态快照：2026-08-08（Asia/Shanghai）
> 适用项目：`Siyuan-Xue/openpi05-bsp` 第一阶段 BSP 复现
> 性质：实际执行记录，不替代[第一阶段服务器 Runbook](pi05_libero_bsp_phase1_server.md)

本文根据服务器终端输出和本次协作记录整理。早期命令由用户在服务器上执行；后期在用户明确授权后，助手通过阿里云 Web VS Code 终端代为执行只读检查和已约定的 pilot 启动/监控。
本文只把有终端证据或用户明确确认的内容记为完成；无法从现有证据确认的内容会标为“未确认”或“尚未执行”。

## 1. 状态标记

| 标记 | 含义 |
| --- | --- |
| 已确认 | 有终端输出或用户明确回报证明已经完成 |
| 临时 | 只在执行过 `export` 的当前 shell 中生效，未证明写入启动文件 |
| 永久配置 | 已明确写入 `~/.bashrc` 或 pip 用户配置 |
| 进行中 | 已启动，但尚未出现成功终态和完整产物 |
| 未确认 | 曾计划或建议执行，但没有完成证据 |
| 尚未执行 | 明确还未进入该阶段 |

本文件位于公开仓库，因此不记录完整主机名、GPU UUID、Kubernetes Pod/container ID、token、credential 或 secret。

## 2. 服务器基线

### 2.1 硬件和基础运行时

| 项目 | 已确认状态 |
| --- | --- |
| 平台 | 阿里云 DSW 托管环境，运行在 Kata/Kubernetes 容器中 |
| CPU | 40 个逻辑 CPU，单 NUMA 节点 |
| 内存 | 128 GiB，检查时约 127 GiB available |
| Swap | 0 B |
| GPU | 单张 NVIDIA H20 |
| GPU 显存 | 97,871 MiB |
| NVIDIA 驱动 | 535.183.06 |
| 驱动报告的 CUDA 上限 | 12.8；这不是要求安装系统 CUDA 12.8 的指令 |
| 基础镜像 Python | 3.12.13；不用于 OpenPI 复现实验 |
| 项目 Python | 独立 CPython 3.11.9 |

### 2.2 容器和 Docker 能力

- Docker CLI 28.1.0 存在。
- Docker daemon 不可访问：`Cannot connect to the Docker daemon at unix:///var/run/docker.sock`。
- `docker compose` 不存在。
- `nvidia-ctk` 和 `nvidia-container-cli` 不存在。
- 因此当前不能使用 runbook 的 Docker + Compose 路线；后续仿真应采用相互隔离的 OpenPI 与 LIBERO 宿主机环境。
- 不因这一限制安装或改动宿主机 Docker、CUDA、驱动或 NVIDIA Container Toolkit。

### 2.3 文件系统

| 路径 | 文件系统和容量 | 本项目用途 | 风险或限制 |
| --- | --- | --- | --- |
| `/root` | 容器 overlay，约 383 GiB；mb8 失败后最后观察已用约 191 GiB、可用约 173 GiB | 仓库、Python 环境、工具、本地 cache、日志和临时探针 checkpoint | 没有证明在 DSW 容器重建后仍然存在，不能直接视为持久盘 |
| `/mnt/workspace` | `/dev/vda` 上约 30 GiB ext4 | 不作为当前项目根目录 | 容量过小，禁止存大模型、LIBERO 数据和正式训练 checkpoint |
| `/mnt/data` | 512 TiB `fuse.ossfs2` | 对象存储挂载入口 | 不是完整的本地 POSIX 块盘；文件锁、原子 rename 和随机 I/O 需要单独验证 |
| `/mnt/data/siyuanxue` | 位于上述 `ossfs2` | 本项目唯一允许写入的数据命名空间 | 所有数据盘写操作必须限制在此路径内 |

`/mnt/data/others` 已不存在。当前规则不是“避开某一个旧目录”，而是：

> `/mnt/data` 下只有 `/mnt/data/siyuanxue` 允许本项目写入；其余所有同级目录均禁止写入、覆盖、重命名、移动或删除。

## 3. 已做的永久配置

### 3.1 `~/.bashrc`

以下内容已明确通过 `>> ~/.bashrc` 写入，并已执行过 `source ~/.bashrc`：

| 配置 | 用途 | 注意事项 |
| --- | --- | --- |
| `export INTRA_CLOUD_ACCELERATION_REGION=cn-wulanchabu` | 启用阿里云内网加速区域设置 | 是否影响某个具体下载器取决于该下载器是否识别此变量 |
| `alias cdw="cd /mnt/workspace"` | 快速进入 `/mnt/workspace` | 该盘只有约 30 GiB，不是大数据或 checkpoint 目录 |
| `alias cdd="cd /mnt/data"` | 快速进入数据盘挂载点 | `/mnt/data` 根目录并非项目可写目录；随后必须进入 `/mnt/data/siyuanxue` |

`cdd` 只是导航别名，绝不表示整个 `/mnt/data` 都可写。

### 3.2 pip 用户配置

以下设置通过 `pip config` 写入 pip 用户配置；默认情况下 root 用户对应的配置文件位于 `/root/.config/pip/pip.conf`，实际位置应以 `pip config debug` 为准：

| 键 | 当前设置 |
| --- | --- |
| `global.index-url` | `http://mirrors.cloud.aliyuncs.com/pypi/simple/` |
| `install.trusted-host` | `mirrors.cloud.aliyuncs.com` |
| `global.extra-index-url` | 已执行 unset |

该镜像使用 HTTP，并通过 `trusted-host` 跳过该主机的 TLS 校验。项目安装仍必须以仓库 lockfile/导出的锁定依赖为准，禁止使用全局、未锁定的 `pip install` 扩展系统环境。

## 4. 临时环境变量

以下变量只在执行过相应 `export` 的 shell 中有效；现有证据没有证明它们已写入 `~/.bashrc`。

### 4.1 当前工作路径

| 变量 | 已采用的值 | 用途 |
| --- | --- | --- |
| `BSP_WORK` | `/root/openpi-bsp-work` | 本地工作根目录 |
| `BSP_REPO_DIR` | `/root/openpi-bsp-work/repo/openpi05-bsp` | 主仓库 |
| `OPENPI_PY` | `/root/openpi-bsp-work/venvs/openpi/bin/python` | 唯一允许用于 OpenPI 的 Python |
| `OPENPI_DATA_HOME` | `/root/openpi-bsp-work/cache/openpi` | OpenPI 官方 checkpoint/tokenizer 的本地 cache |
| `LIBERO_DATASET_DIR` | `/mnt/data/siyuanxue/openpi-bsp/data/lerobot/physical-intelligence/libero` | 官方 LIBERO v2.0 数据根目录 |
| `LOG_BASE` | `/root/openpi-bsp-work/experiments/logs` | 下载、测试和身份日志 |
| `UV_BIN` | `/root/openpi-bsp-work/venvs/uv-0.11.32-bin/uv` | 固定调用独立 uv 二进制 |
| `LIBERO_VENV` | `/root/openpi-bsp-work/venvs/libero-py38` | 与 OpenPI 隔离的 LIBERO simulator 环境 |
| `LIBERO_PY` | `/root/openpi-bsp-work/venvs/libero-py38/bin/python` | 唯一允许用于 LIBERO simulator 的 Python |
| `LIBERO_PY_SOURCE` | `/root/openpi-bsp-work/venvs/python-builds/cpython-3.8-linux-x86_64-gnu/bin/python3.8` | 创建 simulator venv 时使用的 uv 管理解释器 |

### 4.2 Hugging Face 和网络

| 变量 | 已采用的值 | 用途 |
| --- | --- | --- |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 替代当前不可直连的 Hugging Face 官方入口 |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | 禁用 `hf_transfer` 下载后端 |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | `1` | 公共资源请求不隐式携带本地 HF token |
| `HF_HOME` | `/root/openpi-bsp-work/cache/huggingface` | Hugging Face 本地 cache 根目录 |
| `HF_HUB_CACHE` | `/root/openpi-bsp-work/cache/huggingface/hub` | Hugging Face Hub cache |

### 4.3 后台任务变量

| 变量 | 最后采用的值 | 状态 |
| --- | --- | --- |
| `DOWNLOAD_LOG` | `/root/openpi-bsp-work/experiments/logs/libero-v2-download-9047153.log` | 数据集下载已完成；日志仍是审计证据 |
| `DOWNLOAD_PID` | `/root/openpi-bsp-work/experiments/logs/libero-v2-download-9047153.pid` | 进程已结束；PID 文件不是活跃状态证明 |
| `BASE_LOG` | `/root/openpi-bsp-work/experiments/logs/pi05-base-prefetch.log` | 已完成的 `pi05_base` 下载日志 |
| `BASE_PID` | `/root/openpi-bsp-work/experiments/logs/pi05-base-prefetch.pid` | 已完成任务的 PID 记录，不代表仍有活跃进程 |
| `OFFICIAL_LOG` | `/root/openpi-bsp-work/experiments/logs/pi05-libero-gsutil-prefetch.log` | 官方 `pi05_libero` 已完成；此前的 `pi05-libero-prefetch.log` 保留为下载器切换前日志 |
| `OFFICIAL_PID` | `/root/openpi-bsp-work/experiments/logs/pi05-libero-gsutil-prefetch.pid` | 进程已结束；PID 文件不再代表活跃下载 |

### 4.4 安装期间使用过的命令级变量

这些变量主要通过 `env NAME=value command` 传入单次命令，不应假定在新 shell 中存在：

- `UV_CACHE_DIR=/root/openpi-bsp-work/cache/uv`
- `UV_PYTHON_INSTALL_DIR=/root/openpi-bsp-work/venvs/python-builds`
- `UV_PROJECT_ENVIRONMENT=/root/openpi-bsp-work/venvs/openpi`
- `UV_HTTP_TIMEOUT=120`
- `GCLOUD_VERSION=578.0.0`
- `GCLOUD_ARCHIVE=/root/openpi-bsp-work/staging/google-cloud-cli-578.0.0-linux-x86_64.tar.gz`
- `GCLOUD_HOME=/root/openpi-bsp-work/tools/google-cloud-cli-578.0.0`
- `CLOUDSDK_CONFIG=/root/openpi-bsp-work/cache/gcloud`
- `CLOUDSDK_CORE_DISABLE_PROMPTS=1`
- `CLOUDSDK_CORE_DISABLE_USAGE_REPORTING=true`
- `CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=1`

安装 Google Cloud CLI 后曾把 `${GCLOUD_HOME}/bin` 临时前置到 `PATH`。新 shell
不会自动继承该修改；需要使用 `gsutil` 或 `gcloud` 时必须重新显式导出。

### 4.5 后续阶段已使用的临时变量

以下变量已在后续仿真、数据准备、norm stats 或训练探针中实际使用，但仍只属于
当前 shell 或单条 `env` 命令，不是永久配置：

| 变量 | 已采用的值或形式 | 用途 |
| --- | --- | --- |
| `JAX_COMPILATION_CACHE_DIR` | `/root/openpi-bsp-work/cache/jax` | JAX/XLA 编译 cache |
| `WANDB_DIR` | `/root/openpi-bsp-work/experiments/wandb` | W&B 本地日志 |
| `WANDB_MODE` | `offline` | pilot 不向外部服务同步 |
| `CUDA_VISIBLE_DEVICES` | `0` | 固定使用唯一 H20 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 最终探针使用 `0.95`；早期失败尝试使用过 `0.90` | 控制 JAX GPU allocator 预留比例 |
| `BSP_CACHE` | `/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/libero-v2.0-bsp-v2.npz` | 训练读取的持久 BSP sidecar |
| `ASSETS_BASE` | `/root/openpi-bsp-work/experiments/assets` | baseline/BSP norm stats |
| `PROBE_CHECKPOINT_BASE` | `/root/openpi-bsp-work/experiments/checkpoints/probes` | 单步显存探针 checkpoint |
| pilot checkpoint base | `/root/openpi-bsp-work/experiments/checkpoints/pilots` | 10/100-step 稳定性 pilot |
| `EVAL_BASE` | `/root/openpi-bsp-work/experiments/eval` | 仿真评测产物 |
| `LIBERO_CONFIG_PATH` | `/root/openpi-bsp-work/cache/libero-config` | LIBERO 用户配置隔离路径 |
| `LIBERO_PYTHONPATH` | 仓库根、`packages/openpi-client/src`、`third_party/libero` | 解决两个 editable 源码包的运行时导入路径 |
| `EGL_VENDOR_JSON` | `/root/openpi-bsp-work/cache/egl/10_nvidia.json` | 私有 GLVND NVIDIA EGL vendor 描述 |

`PYTHONNOUSERSITE=1` 仍属于 runbook 建议，现有证据没有证明它已用于全部命令。

仅讨论过但没有终端证据证明已配置的网络项包括 `HTTP_PROXY`、`HTTPS_PROXY`、
`ALL_PROXY`、Google Cloud CLI proxy properties、阿里云 Global Accelerator，以及
VPC NAT Gateway/EIP/SNAT。不得把网络配置建议记成服务器现状。

进入相应阶段时应重新显式导出，不得依赖旧 shell。

## 5. 目录结构和职责

状态列基于对话中的创建命令、路径输出和后续实际使用。

```text
/root/openpi-bsp-work/
  repo/
    openpi05-bsp/                      # [已确认] 主仓库
      third_party/libero/              # [已确认] 锁定的 LIBERO 子模块
  venvs/
    uv-0.11.32-bin/                    # [已确认] 独立 uv 二进制
    python-builds/
      cpython-3.11.9-linux-x86_64-gnu/ # [已确认] uv 管理的 CPython
      cpython-3.8-linux-x86_64-gnu/    # [已确认] uv 管理的 CPython 3.8.20
    openpi/                            # [已确认] OpenPI Python 3.11.9 环境
    libero-py38/                       # [已确认] Python 3.8.20 LIBERO 仿真环境及锁定依赖
  cache/
    uv/                               # [已确认] uv 下载/构建 cache
    huggingface/
      hub/                            # [已确认] Hugging Face Hub cache 目录
    gcloud/                           # [已确认] Google Cloud CLI 本地配置/cache
    egl/
      10_nvidia.json                 # [已确认] 私有 NVIDIA EGL vendor 描述
    libero-config/                   # [已确认] LIBERO 用户配置隔离目录
    openpi/                           # [已确认] OpenPI 官方资产 cache
      openpi-assets/checkpoints/
        pi05_base/                    # [已确认] 官方 π0.5 基座 checkpoint
        pi05_libero/                  # [已确认] 官方 LIBERO 校准 checkpoint
      big_vision/
        paligemma_tokenizer.model     # [已确认] PaliGemma tokenizer
  staging/
    openpi-locked.txt                 # [已确认] 镜像安装用的锁定依赖中间文件
    bspline-targets/
      libero-v2.0-bsp-v2.npz         # [已确认] sidecar 本地构建/验证副本
    google-cloud-cli-578.0.0-linux-x86_64.tar.gz
                                       # [已确认] 完整性已核验的 Cloud CLI 归档
  tools/
    google-cloud-cli-578.0.0/         # [已确认] 隔离的 gcloud/gsutil 工具目录
  experiments/
    logs/                             # [已确认] 下载、测试、哈希和环境日志
    assets/                           # [已确认] baseline/BSP norm stats 与比较 JSON
    checkpoints/
      probes/                        # [已确认] 一步显存探针；部分旧探针已清理
      pilots/                        # [已确认] 10/100-step pilot 路径；失败运行无有效 checkpoint
    eval/                             # [已确认] EGL 冒烟与官方 200-rollout 校准结果
    wandb/                            # [已确认] 离线 W&B 运行日志
```

```text
/mnt/data/
  siyuanxue/                          # 本项目唯一允许写入的命名空间
    openpi-bsp/
      data/
        lerobot/
          physical-intelligence/
            libero/                   # [已确认] 官方 LIBERO v2.0
        bspline-targets/              # [已确认] BSP sidecar 与 verification
          libero-v2.0-bsp-v2.npz     # [已确认] 持久 sidecar
          libero-v2.0-bsp-v2.verification.json
                                       # [已确认] 持久 verification 诊断
      experiments/
        assets-archives/              # [已确认] norm 资产审计归档
        eval-archives/                # [预留；是否已完成复制未确认] 评测审计归档
  <其他目录>/                          # 非本项目目录，全部禁止写入
```

目录职责：

- `repo/openpi05-bsp`：唯一需要部署到服务器的源码仓库。
- `third_party/libero`：仿真代码和任务资产的固定版本，不是训练数据副本。
- `venvs/openpi`：OpenPI 训练和 policy server 的 Python 3.11.9 环境。
- `venvs/libero-py38`：LIBERO simulator 的 Python 3.8.20 环境；依赖已安装并通过真实 EGL
  reset/render/step，仍不得与 OpenPI 3.11 环境混用。
- `cache/openpi`：可重新下载、可跨 baseline/BSP 复用的官方模型和 tokenizer，不是训练产物。
- `tools/google-cloud-cli-578.0.0`：为 GCS 大文件并行下载部署的隔离工具，不是系统安装。
- `experiments/assets`：已生成 baseline/BSP 独立 norm stats 和状态一致性比较结果。
- `experiments/checkpoints`：探针/pilot 产物；正式训练前仍必须解决显存协议以及容量和持久性问题。
- `experiments/eval`：已生成逐回合 JSONL、CSV、汇总 JSON 和选定视频。
- `/mnt/data/siyuanxue/openpi-bsp/data`：持久训练数据、BSP sidecar 和 verification；不在这里放仓库或 Python 环境。

## 6. 已写入资源和身份

### 6.1 仓库和子模块

| 资源 | 已确认身份 |
| --- | --- |
| 主仓库 | `https://github.com/Siyuan-Xue/openpi05-bsp.git` |
| 当前主仓库 commit | `904715355b396715781fc3aa3d1cebbee0890273` |
| 分支状态 | `main` 跟踪 `origin/main`；最后确认工作区干净 |
| LIBERO 子模块 commit | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |

只初始化了 `third_party/libero`。`third_party/aloha` 不属于本阶段，不应初始化。

### 6.2 Python 和依赖环境

| 资源 | 已确认状态 |
| --- | --- |
| uv | 0.11.32，独立安装在项目工作目录 |
| CPython | 3.11.9，由 uv 安装到 `venvs/python-builds` |
| OpenPI venv | `/root/openpi-bsp-work/venvs/openpi` |
| LIBERO CPython | 3.8.20，由 uv 安装到 `venvs/python-builds/cpython-3.8-linux-x86_64-gnu` |
| LIBERO venv | `/root/openpi-bsp-work/venvs/libero-py38`；解释器、依赖、源码包和真实 EGL 仿真门禁通过 |
| SciPy | 1.15.3，按 BSP cache fingerprint 协议锁定 |
| JAX | 已成功列出 `[CudaDevice(id=0)]` |
| LeRobot | 仓库 lockfile 锁定的 git revision，由 OpenPI venv 安装 |

最初直接使用系统 `pip` 的审计显示：系统 Python 为 3.12.13，且系统环境中没有安装 `trimesh`。这不是缺陷；项目依赖必须存在于独立 OpenPI/LIBERO 环境中，而不是全局环境。

创建 LIBERO venv 时 uv 提示仓库根项目要求 Python `>=3.11`。这是因为命令在 OpenPI
workspace 中执行，而不是 venv 创建失败；显式目标解释器为 3.8.20，终端已经确认
`prefix=/root/openpi-bsp-work/venvs/libero-py38`、独立 3.8 `base_prefix` 和
`libero_python38=PASS`。

后续环境身份已确认：NumPy 1.22.4、PyTorch 1.11.0+cu113、MuJoCo 3.2.3、
robosuite 1.4.1、LIBERO 0.1.0、openpi-client 0.1.0。`libero` 和 `openpi-client`
均以 editable 方式指向锁定仓库源码；由于该 LIBERO revision 的源码布局，单凭
editable 元数据不足以保证 `import libero`，实际 evaluator 使用显式
`LIBERO_PYTHONPATH`，其中包含仓库根、`packages/openpi-client/src` 和
`third_party/libero`。`libero-py38-freeze.txt` 已生成并通过非空门禁。

### 6.3 LIBERO 数据集

| 项目 | 已确认值 |
| --- | --- |
| Repo ID | `physical-intelligence/libero` |
| Revision | `v2.0` |
| Episodes | 1,693 |
| Frames | 273,465 |
| Tasks | 40 |
| FPS | 10 |
| 历史下载日志 fingerprint | `9bee9f57a76f0915f90fe258660d064447e3205f746e094b095d1106f8c4ae40` |
| 当前准备/sidecar manifest fingerprint | `db8fe671f0e0ad33dcf2ef2e563c779c0f6c2cc4d91e314379d1c0bc64768213` |
| Hugging Face dataset fingerprint | `de4a79e770bcac3f` |
| Episode boundaries SHA-256 | `749c41e05d2f336c6f37c309c7700d4ea748680bb18c049d1978b8787c73c351` |
| 持久位置 | `/mnt/data/siyuanxue/openpi-bsp/data/lerobot/physical-intelligence/libero` |

下载日志确认 1,699 个仓库文件被取回；该数字是文件数，不是 episode 数。LeRobot 输出的 v2.0/global-stats 提示是兼容性 warning，不是失败。

早期下载命令和当前 sidecar 代码产生了两个不同的顶层 fingerprint。当前代码连续两次
只读 recheck 均复现 `db8fe671...`，sidecar manifest 也使用同一值，因此后续 BSP
cache/norm/训练身份以 `db8fe671...` 为准；`9bee9f57...` 仅保留为历史下载日志证据，
不能与当前 cache fingerprint 混用。现有证据没有表明数据内容被转换为 v2.1。

禁止运行 LeRobot 提示的 v2.0 到 v2.1 转换命令。本实验固定使用 v2.0；转换会改变数据格式和 fingerprint。

### 6.4 `pi05_base`

状态已确认完成：

- 目标 URI：`gs://openpi-assets/checkpoints/pi05_base`
- 最终目录：`/root/openpi-bsp-work/cache/openpi/openpi-assets/checkpoints/pi05_base`
- `du -sh` 的最终显示为 12 GiB。
- `pi05_base/params` 目录检查通过，终端输出 `pi05_base=PASS`。
- 官方 GCS 对象清单共 29 个对象、12,441,749,581 bytes，即约 12.44 GB / 11.59 GiB。
- 下载当时服务器还没有 `gsutil`，OpenPI 自动回退到低并发 `gcsfs`；速度慢本身不表示失败。
- 完整目录树 SHA-256：`29ad861de329d3383efe74052273720f9efb6f607cb4bf86de740198d9cc2518`。
- 身份日志：`experiments/logs/pi05-base-tree.sha256`，格式门禁输出 `pi05_base_tree_hash=PASS`。

该目录树摘要覆盖相对路径和每个普通文件的内容，用于证明 baseline/BSP 从同一份
基座权重开始；它不是 Google 发布的官方整目录校验值。

### 6.5 官方 `pi05_libero`

状态已确认完成：

- 目标 URI：`gs://openpi-assets/checkpoints/pi05_libero`
- 最终目录：`/root/openpi-bsp-work/cache/openpi/openpi-assets/checkpoints/pi05_libero`
- 官方 GCS 对象清单共 16 个对象、12,439,085,481 bytes，即约 12.44 GB / 11.58 GiB。
- 低并发 `gcsfs` 阶段最后记录过 249,366,180 bytes，即约 0.23 GiB / 11.58 GiB（2.00%）；随后停止该单进程下载，部署隔离的 Google Cloud CLI，并用 `gsutil -m` 路线继续。
- `du -sh` 的最终显示为 12 GiB，`pi05_libero/params` 存在，终端输出 `pi05_libero=PASS`。
- 原 `.partial` 已不存在，门禁输出 `partial_removed=PASS`。
- checkpoint 内含 `assets/physical-intelligence/libero/norm_stats.json`，大小 1,914 bytes。
- 官方 norm stats SHA-256：`b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`。
- 完整目录树 SHA-256：`42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99`。
- 身份日志包括 `official-norm.sha256` 和 `pi05-libero-tree.sha256`；目录树格式门禁输出 `pi05_libero_tree_hash=PASS`。

一次对不存在的 `pi05-libero-remote-list.txt` 执行 `grep` 曾输出
`remote_asset_match=NONE`。该结果只是输入日志文件未生成，不能解释为远端缺少资产；
随后完整本地目录清单已经直接证明 norm stats 存在。

### 6.6 PaliGemma tokenizer

状态已确认完成：

- URI：`gs://big_vision/paligemma_tokenizer.model`
- 本地路径：`/root/openpi-bsp-work/cache/openpi/big_vision/paligemma_tokenizer.model`
- 大小：4,264,023 bytes。
- SHA-256：`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。
- 非空门禁输出 `paligemma_tokenizer=PASS`，下载与哈希日志均已写入 `experiments/logs`。

### 6.7 Google Cloud CLI 和 GCS 下载工具

- 版本：578.0.0，使用独立归档，不通过系统包管理器安装。
- 归档路径：`/root/openpi-bsp-work/staging/google-cloud-cli-578.0.0-linux-x86_64.tar.gz`。
- 归档大小：88,566,204 bytes；内部 `VERSION` 文件为 `578.0.0`，`tar -tzf` 返回 0。
- 本地 SHA-256：`322ac42ef7670cf2e16d46a1c3f827b36e55a865d2e26f34c64c914869e400f0`。
- 本地 MD5：`4ba75048ec11ca9343dbe1edfdbcec92`；其十六进制和 Base64 形式均与 GCS 官方响应的 ETag/MD5 完全一致。
- 工具目录：`/root/openpi-bsp-work/tools/google-cloud-cli-578.0.0`。
- Google Cloud CLI 只通过临时 `PATH` 和独立 `CLOUDSDK_CONFIG` 使用；未写入系统目录。

最初提供的一个 SHA-256 期望值与归档不匹配。没有据此删除归档；后续通过精确字节数、
tar 完整性、内部版本以及官方远端 MD5 四项证据确认归档有效。这是一次校验值来源错误，
不是归档损坏。

### 6.8 BSP sidecar 与 norm stats

状态已确认完成：

- 本地构建副本：`/root/openpi-bsp-work/staging/bspline-targets/libero-v2.0-bsp-v2.npz`，
  `ls -lh` 约 8.1 MiB。
- 持久 sidecar：`/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/libero-v2.0-bsp-v2.npz`。
- 持久 verification：`/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/libero-v2.0-bsp-v2.verification.json`。
- `targets` 为 `(259121, 16, 8)`、`float32`；`mapping` 为 `(273465,)`、`uint32`，
  所有 target 有限，最大 mapping 小于 target 数。
- manifest format version 为 2，fingerprint 为 `db8fe671...`；全量 verify 已结束并由
  用户确认通过。
- 协议固定：degree 3、chunk size 10、16 rows、8 channels、7D controls + 1 knot、
  max absolute error 0.002、smoothing `1e-12`、stride 1、frame-index 时间轴、
  episode-start 缓存 knot、materialize 时转为当前 episode-local frame、解码 8 动作。

两套 quantile norm stats 已分别扫描 1,068 个 batch，各耗时约 40 分钟：

| 产物 | SHA-256 |
| --- | --- |
| baseline `norm_stats.json` | `2c37ee75029e8d01f36210e187ed567ec8860961c795b60522e12c82e86d7635` |
| BSP `norm_stats.json` | `00e1477be94d3106ff2803acf244eee6a623c96bb0ef91a3be43af6e92671a32` |
| `libero-phase1-norm-comparison.json` | `d654463a9e1652fb90380a31fde1c6885a7437d0dd8b15b35be0bf9aff2892d2` |

比较门禁确认 `state_stats_equal=true`、`asset_directories_isolated=true`、
`action_stats_isolated=true`；state 的 mean/std/q01/q99 均逐项相等且最大绝对差为 0，
baseline/BSP action stats 摘要不同。审计归档
`/mnt/data/siyuanxue/openpi-bsp/experiments/assets-archives/phase1-norm-assets-db8fe671.tar`
及其 `.sha256` 最终由用户确认全部通过。首次归档因新 shell 中 `ASSETS_BASE` 为空而
生成 0-byte 非法 tar；该失败被识别，没有把它当作有效归档。

### 6.9 尚未完成的核心实验资源

- 可连续运行的 baseline/BSP 100-step pilot；当前全量微调即使无 EMA、micro-batch 8
  仍在第二个 optimizer step OOM。
- 两组 30k 正式训练和 0k/5k/10k/20k/30k 十个固定 checkpoint。
- 十个 checkpoint 的四套件 × 50 rollouts（合计 20,000 回合）及 bootstrap 统计报告。
- 正式 checkpoint 的可靠持久化方案；`/root` 仍未证明跨 DSW 重建持久，直接写
  `ossfs2` 的 Orbax 语义也未验证。

## 7. 已执行测试和修复历史

### 7.1 已确认通过

- 运行时与训练规划合同：21 passed。
- BSP、数据适配、data loader、LIBERO policy 和准备脚本测试：55 passed。
- 上述 55 个测试出现一条 `ml_collections` 第三方 `DeprecationWarning`，不影响通过状态。
- `prepare_libero_bsp_test.py` 的真实 Tyro CLI 回归测试修正后，用户回报测试通过；当前预期计数为 5 passed。
- baseline/BSP 训练帮助页已成功生成到 `experiments/logs/help/`；已确认训练 CLI 暴露
  `--micro-batch-size`、`--data.lerobot-root`、`--data.bsp-cache-path` 和
  `--checkpoint-base-dir`，与第一阶段运行协议一致。
- norm stats、数据准备和推理服务帮助页已成功生成；已确认 norm CLI 暴露
  `--assets-dir`、`--bsp-cache-path` 和 `--dataset-root`，准备 CLI 暴露
  `download`、`build`、`verify` 三种模式，推理服务暴露 `policy:checkpoint`。
- evaluator 和第一阶段比较器帮助页也已确认符合预期，包括
  `--args.task-suite-name`、`--args.checkpoint-step`、`--bsp-verification`、
  `--norm-comparison` 和 `--output-dir`；首次重任务前的 CLI 帮助页门禁已完整通过。
- 最终 commit 上补跑 `train_test.py`、服务端 evaluator 测试及客户端 evaluation/report
  测试，结果为 46 passed、338 warnings、返回码 0。338 是重复发出次数，日志归并为
  9 条不同消息：4 条 Beartype 对 Flax/Optax 展开类型别名的 PEP 585 弃用提示、
  2 条锁定 JAX/Flax 栈的未来兼容弃用提示、2 条 JAX buffer donation 未生效提示，
  以及 1 条 Orbax 缺少显式 restore sharding 的提示。当前锁定 Python/JAX 环境下
  不阻塞继续执行；后两类分别在显存探针和 checkpoint restore/pilot 中继续审计。
- 宿主机动态库门禁已确认 `libEGL.so.1`、`libGL.so.1`、NVIDIA EGL/GLX 实现及
  `libEGL_nvidia.so.0` 的依赖均完整。DSW 容器缺少系统 GLVND vendor JSON，项目在
  `cache/egl/10_nvidia.json` 写入最小 NVIDIA ICD 描述，并通过
  `__EGL_VENDOR_LIBRARY_FILENAMES` 仅对仿真进程启用；MuJoCo 3.2.3 的 16×16 EGL
  context 创建、make-current 和释放均通过。
- LIBERO `libero_spatial` task 0 的 reset/render/dummy-step 门禁通过：agentview 与 wrist
  均为 `(256, 256, 3)`、`uint8`、有限值，保存了方向校正 PNG。
- 官方 `pi05_libero@30k` policy server 的 `/healthz` 返回 200/OK；checkpoint 恢复约
  6.2 GiB 参数，日志确认 norm stats 从 checkpoint 自带 assets 目录加载。
- 官方 checkpoint 单回合闭环冒烟通过：`libero_spatial` task 0、init 0，76 steps 成功，
  JSONL/manifest/CSV/summary 产物完整，无 infrastructure 或 artifact error。
- 四套件 × 10 tasks × 5 initial states 的 200-rollout 校准通过验收：196/200 成功，
  suite macro success rate 0.98；`libero_10=48/50`、`libero_goal=48/50`、
  `libero_object=50/50`、`libero_spatial=50/50`，200 个 episode 全部 eligible，
  `acceptance_complete=true`，基础设施和产物错误均为 0。结果目录约 3.9 MiB，含
  44 个 MP4；这是环境校准，不是 baseline/BSP 第一阶段正式对比结果。
- 身份复核仍为 commit `904715355b396715781fc3aa3d1cebbee0890273`、干净的
  `main...origin/main`、LIBERO 子模块 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`、
  Python 3.11.9、SciPy 1.15.3，且 JAX 识别一个 CUDA device。

### 7.2 曾出现且已定位的问题

1. `scripts/train_test.py` 曾比较两个独立随机训练过程的参数是否逐位相等，63/64 个元素出现约 `5e-10` 的差异。两个配置实际解析为同一训练计划，问题属于错误测试假设，不是梯度累积实现失败。该测试在 commit `c06adbe83c33571bb8fd019950f4077f87b5d7a3` 中移除。
2. 数据准备 CLI 最初只接受大写枚举名，但 runbook 使用小写 `download/build/verify`。生产注解改为 Tyro enum values 后，命令可接受小写。
3. 首版回归测试误用 `tyro.extras.get_parser().parse_args()`，拿到的是 argparse 原始列表而非 Tyro 最终枚举。当前 commit `904715355b396715781fc3aa3d1cebbee0890273` 改为调用真实 `tyro.cli(...)`。
4. 初次 `import robosuite` 因 DSW 没有 GLVND vendor JSON 而报
   `Cannot initialize a EGL device display`。GPU 设备节点和 NVIDIA EGL 动态库均存在，
   根因是 `/usr/share/glvnd/egl_vendor.d` 与 `/etc/glvnd/egl_vendor.d` 均缺失；采用
   项目私有 `10_nvidia.json` 后真实 EGL context 通过，没有安装系统包或修改系统目录。
5. `uv pip show libero` 能看到 editable 元数据，但 `import libero` 一度失败；原因是锁定
   LIBERO revision 的包布局，不是 wheel 丢失。显式 `PYTHONPATH` 指向锁定源码后 import
   和完整 evaluator 均通过。

现在已经有最终 commit 上重跑 `scripts/train_test.py` 的通过证据。不得为消除弃用
warning 而升级锁定依赖；JAX donation warning 若在正式探针中持续出现，按实际峰值
显存选共同 micro-batch；Orbax warning 则要求在相同单 GPU 拓扑完成真实加载、保存和
恢复门禁，不能仅凭单元测试返回码推断 checkpoint 可跨拓扑恢复。

### 7.3 补充合同测试

以下轻量合同测试已确认通过：

```bash
"$OPENPI_PY" -m pytest -q \
  scripts/libero_compose_preflight_test.py \
  scripts/compare_libero_phase1_test.py
```

服务器输出为 `13 passed in 0.06s`。

## 8. 网络状态

| 端点 | 已确认状态 | 当前策略 |
| --- | --- | --- |
| GitHub | 曾发生一次 GnuTLS 非正常终止；之后 `ls-remote`、clone 和 pull 成功 | 失败时先重试只读请求，不改 git 历史 |
| Hugging Face 官方站 | 443 连接被拒绝 | 使用 `HF_ENDPOINT=https://hf-mirror.com` |
| Hugging Face 镜像 | API 实际 GET 返回 200，完整 LIBERO 已下载 | 禁用 `hf_transfer`，保持 v2.0 revision |
| GCS | 存储 API 路径可达；`pi05_base`、`pi05_libero` 和 tokenizer 均已完整下载 | `pi05_base` 使用 `gcsfs`；后续安装隔离 Google Cloud CLI，并通过 `gsutil -m` 完成官方 LIBERO checkpoint |
| 阿里云 PyPI | simple index 返回 200，pip 查询成功 | 只安装锁定依赖到独立 venv |
| Docker Hub | 连接超时 | 当前不采用 Docker 路线 |
| GHCR | HTTP 路径可达，但没有执行镜像拉取 | 不把可达性等同于可用镜像环境 |

没有终端证据表明服务器已配置命令行代理、Global Accelerator 或专用 NAT/EIP/SNAT。
相关讨论只是网络加速备选方案。现有官方资产已经下载完成，不应为了“优化”已完成任务
而改变全局代理或下载路径。

## 9. 新 shell 初始化

下面的命令只恢复当前已经采用的变量，不写入系统配置：

```bash
export BSP_WORK=/root/openpi-bsp-work
export BSP_REPO_DIR="$BSP_WORK/repo/openpi05-bsp"
export OPENPI_PY="$BSP_WORK/venvs/openpi/bin/python"
export OPENPI_DATA_HOME="$BSP_WORK/cache/openpi"
export LOG_BASE="$BSP_WORK/experiments/logs"
export ASSETS_BASE="$BSP_WORK/experiments/assets"
export WANDB_DIR="$BSP_WORK/experiments/wandb"
export JAX_COMPILATION_CACHE_DIR="$BSP_WORK/cache/jax"

export LIBERO_DATASET_DIR=/mnt/data/siyuanxue/openpi-bsp/data/lerobot/physical-intelligence/libero
export BSP_CACHE=/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/libero-v2.0-bsp-v2.npz
export BSP_VERIFY=/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/libero-v2.0-bsp-v2.verification.json

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$BSP_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

export DOWNLOAD_LOG="$LOG_BASE/libero-v2-download-9047153.log"
export DOWNLOAD_PID="$LOG_BASE/libero-v2-download-9047153.pid"
export BASE_LOG="$LOG_BASE/pi05-base-prefetch.log"
export BASE_PID="$LOG_BASE/pi05-base-prefetch.pid"
export OFFICIAL_LOG="$LOG_BASE/pi05-libero-gsutil-prefetch.log"
export OFFICIAL_PID="$LOG_BASE/pi05-libero-gsutil-prefetch.pid"
export OFFICIAL_DIR="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"

export UV_BIN="$BSP_WORK/venvs/uv-0.11.32-bin/uv"
export UV_CACHE_DIR="$BSP_WORK/cache/uv"
export UV_PYTHON_INSTALL_DIR="$BSP_WORK/venvs/python-builds"
export LIBERO_VENV="$BSP_WORK/venvs/libero-py38"
export LIBERO_PY="$LIBERO_VENV/bin/python"
export LIBERO_CONFIG_PATH="$BSP_WORK/cache/libero-config"
export LIBERO_PYTHONPATH="$BSP_REPO_DIR:$BSP_REPO_DIR/packages/openpi-client/src:$BSP_REPO_DIR/third_party/libero"
export EGL_VENDOR_JSON="$BSP_WORK/cache/egl/10_nvidia.json"

export GCLOUD_HOME="$BSP_WORK/tools/google-cloud-cli-578.0.0"
export CLOUDSDK_CONFIG="$BSP_WORK/cache/gcloud"
export PATH="$GCLOUD_HOME/bin:$PATH"

export BASE_CKPT_LOCAL="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base"
export OFFICIAL_CKPT_LOCAL="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
export PALIGEMMA_TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
```

不要把这一段追加到 `~/.bashrc`，除非先单独审查每个路径和持久性假设。

## 10. 每次登录后的只读核验

### 10.1 仓库和子模块

```bash
git -C "$BSP_REPO_DIR" rev-parse HEAD
git -C "$BSP_REPO_DIR" status --short --branch
git -C "$BSP_REPO_DIR" submodule status third_party/libero
```

预期主仓库 SHA：

```text
904715355b396715781fc3aa3d1cebbee0890273
```

预期子模块前缀不带 `-` 或 `+`，并指向：

```text
f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

### 10.2 Python 和 GPU

```bash
"$OPENPI_PY" -c 'import platform, scipy; print(platform.python_version(), scipy.__version__)'
"$OPENPI_PY" -c 'import jax; print(jax.devices())'
```

预期分别包含 `3.11.9 1.15.3` 和 `CudaDevice(id=0)`。

### 10.3 数据盘边界

这些命令只读取获准目录的身份，不列举其他目录：

```bash
findmnt -T /mnt/data/siyuanxue -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,OPTIONS
realpath /mnt/data/siyuanxue
test -w /mnt/data/siyuanxue && echo siyuanxue_write_root=PASS
```

对任何准备写入的数据盘路径，先执行以下边界检查：

```bash
assert_siyuanxue_target() {
  local target
  target="$(realpath -m -- "${1:?target path is required}")" || return
  case "$target" in
    /mnt/data/siyuanxue|/mnt/data/siyuanxue/*)
      printf 'allowed_target=%s\n' "$target"
      ;;
    *)
      printf 'STOP: forbidden /mnt/data target: %s\n' "$target" >&2
      return 2
      ;;
  esac
}

assert_siyuanxue_target "$LIBERO_DATASET_DIR"
```

该函数只验证路径，不执行创建、覆盖、移动或删除。

### 10.4 LIBERO 数据身份

避免为了日常检查反复扫描 273,465 帧。优先读取已经生成的下载日志：

```bash
grep -E \
  'Downloaded and validated|Metadata:|Snapshot/cache fingerprint:' \
  "$DOWNLOAD_LOG"
```

预期包含 1,693 episodes、273,465 frames、40 tasks、10 fps；该历史日志会显示
`9bee9f57...`。sidecar/训练身份应另外从 manifest 或 recheck 日志读取，并要求
`db8fe671...`，不要把两个 fingerprint 当成同一个字段。

### 10.5 官方模型和 tokenizer 状态

```bash
if test -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"; then
  echo pi05_base=COMPLETE
elif test -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base.partial"; then
  echo pi05_base=PARTIAL
  du -sb "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base.partial"
else
  echo pi05_base=MISSING
fi

test -f "$BASE_LOG" && tail -n 30 "$BASE_LOG"
```

PID 文件可能在进程结束后保留，也可能对应已经被系统复用的 PID，因此不能单独证明任务仍在运行。需要检查时同时核对命令行：

```bash
test -f "$BASE_PID" && ps -wwfp "$(cat "$BASE_PID")"
```

对官方 LIBERO checkpoint 和 tokenizer 使用同样的最终产物门禁：

```bash
test -d "$OFFICIAL_CKPT_LOCAL/params" && echo pi05_libero=COMPLETE
test ! -e "$OFFICIAL_CKPT_LOCAL.partial" && echo pi05_libero_partial=ABSENT
test -s "$OFFICIAL_CKPT_LOCAL/assets/physical-intelligence/libero/norm_stats.json" \
  && echo official_norm=PASS
test -s "$PALIGEMMA_TOKENIZER" && echo paligemma_tokenizer=PASS

cat \
  "$LOG_BASE/pi05-base-tree.sha256" \
  "$LOG_BASE/pi05-libero-tree.sha256" \
  "$LOG_BASE/official-norm.sha256" \
  "$LOG_BASE/paligemma-tokenizer.sha256"
```

预期目录树身份分别为：

```text
pi05_base 29ad861de329d3383efe74052273720f9efb6f607cb4bf86de740198d9cc2518
pi05_libero 42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99
```

### 10.6 LIBERO Python 隔离状态

```bash
"$LIBERO_PY" -c \
  'import platform, sys; print(platform.python_version()); print(sys.prefix); print(sys.base_prefix)'

test "$("$LIBERO_PY" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = 3.8 \
  && echo libero_python38=PASS
```

依赖身份和运行时导入使用：

```bash
env \
  PYTHONPATH="$LIBERO_PYTHONPATH" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR_JSON" \
  "$LIBERO_PY" -c \
  'import numpy, torch, mujoco, robosuite, libero, openpi_client; print("libero_runtime=PASS")'
```

robosuite 的 private macro 提示、NumPy/Numba 弃用提示、LIBERO 默认 datasets 路径不存在
提示在已通过的 smoke/calibration 中均为非阻塞 warning。真正门禁仍是 EGL context、
reset/render/step、policy health、逐回合产物和错误分类，而不是“没有 warning”。

### 10.7 永久配置审计

```bash
grep -nE \
  'INTRA_CLOUD_ACCELERATION_REGION|alias cdw=|alias cdd=' \
  /root/.bashrc

python3 -m pip config debug
python3 -m pip config list
```

这些命令不得扩展为打印 token、credential、Kubernetes secret 或云平台凭据文件。

## 11. 服务器使用禁忌

### 11.1 系统环境

- 禁止使用或污染系统 Python 3.12；项目命令统一调用 `$OPENPI_PY` 或后续独立的 LIBERO Python。
- 禁止运行全局 `pip install`；尤其不要在 `/usr/local/lib/python3.12` 中安装项目依赖。
- 禁止安装、升级或替换系统 CUDA、NVIDIA 驱动、系统 PyTorch。
- Docker daemon 不可用时，禁止反复尝试 Docker/Compose 或安装系统 Docker/NVIDIA toolkit。

### 11.2 `/mnt/data` 写入边界

- `/mnt/data/others` 已不存在，不再把它作为现存路径或安全边界。
- `/mnt/data/siyuanxue` 是 `/mnt/data` 下本项目唯一允许写入的位置。
- 严禁向 `/mnt/data` 下除 `siyuanxue` 外的任何目录写入、覆盖、重命名、移动或删除内容。
- 使用 `cdd` 后必须继续进入 `/mnt/data/siyuanxue`；不得在 `/mnt/data` 根目录创建文件。
- 禁止在 `/mnt/data` 中使用未经限定的 `*`、递归删除或以当前目录为目标的写操作。
- 任一数据盘写操作前必须解析并检查目标，确认其等于 `/mnt/data/siyuanxue` 或位于其子目录。

### 11.3 存储和 checkpoint

- 禁止把大数据、模型或正式 checkpoint 写入只有约 30 GiB 的 `/mnt/workspace`。
- 禁止把 `/root` 当作已证明持久盘；正式训练前必须解决训练 checkpoint 的持久性和容量门禁。
- 禁止未经验证就把 Orbax/JAX checkpoint 直接写到 `ossfs2`。必须先证明文件锁、原子 rename、目录语义和随机 I/O 满足要求。
- 训练 checkpoint、评测结果和身份 manifest 是不可重新下载的实验产物，不能和可重建 cache 混为一谈。

### 11.4 数据和源码协议

- 禁止将 LIBERO v2.0 转换为 v2.1。
- 禁止使用 `modified_libero_rlds` 或重新下载原始 RLDS 转换数据。
- 禁止跨 episode 拟合 BSP，禁止在归一化后拟合，禁止改变固定 BSP 参数。
- 禁止初始化 ALOHA 子模块或将 LIBERO 子模块更新到浮动 HEAD。
- 服务器不需要 BSP 作者仓库或论文 PDF，禁止额外部署它们作为运行依赖。
- 禁止执行 `git reset --hard`、`git clean` 或删除未核实的工作区/实验目录。

### 11.5 下载和并行任务

- 禁止删除仍在增长的 `.partial` 目录。
- 禁止在下载过程中修改 `OPENPI_DATA_HOME`、启动同一资产的第二个下载进程或强制重启。
- 任一大模型下载期间，不并发启动另一个 checkpoint 或 tokenizer 下载。
- 大模型下载期间不并发运行 BSP sidecar、norm stats、仿真环境安装、训练或其他重 I/O 工作。
- 禁止在 Kata/DSW 容器中运行来源不明的一键代理脚本或擅自启用 TUN、修改全局路由/DNS。
- 若以后使用获准代理，不得把用户名、密码、订阅链接或 token 写入本文件、shell 历史、
  `~/.bashrc` 或公开日志；代理/GA/NAT 的实际启用状态必须另行留存脱敏证据。
- warning 不等于失败，后台进程消失也不等于成功；必须检查日志终态和完整产物结构。

### 11.6 实验公平性

- 第一阶段固定 seed 42、有效 batch 256、30,000 optimizer steps 和同一 `pi05_base` 初始化。
- micro-batch 只用于梯度累积，不得改变有效 batch。
- 全量 baseline 或 BSP 在共同 micro-batch 2 的持续训练门禁仍失败时，必须停止全量路线并报告；只允许切换到仓库中独立注册、经过合同测试的官方 JAX LoRA 配置，禁止临时改模型或降低有效 batch。
- 当前 pilot 经用户明确选择使用 `ema_decay=None`；正式报告必须标明它与原计划 EMA 协议
  不同，不能把“可运行”解释成“效果等价”。
- 禁止挑选“最好 checkpoint”；固定评测 0k、5k、10k、20k、30k。
- 禁止在第一阶段增加额外 reconstruction、smoothness、monotonicity loss 或 2×/4× 加速。

## 12. 当前下一步

仿真、官方 checkpoint 校准、sidecar、verification 和两套 norm stats 已完成。当前唯一
阻止进入正式训练的直接技术门禁是：全量微调在单张 H20 上无法连续稳定运行既定协议。

下一轮已经由用户批准，必须严格按以下顺序进行：

1. 保留现有失败日志和路径，不重试同名实验、不删除 checkpoint。全量路线依次运行
   baseline/BSP 的 micro-batch 4、10-step 和 100-step 门禁；四项全部通过就选择 4，
   不再为了寻找更小数值而运行 2。
2. 任一 micro-batch 4 阶段失败时，baseline/BSP 都从同一 `pi05_base` 重新按完整
   10/100-step 序列验证 micro-batch 2，不能只补跑失败的一组。有效 batch 始终为 256。
3. micro-batch 2 仍失败时，全量路线判为当前 H20 硬件阻塞，正式候选切换为独立的
   `pi05_libero_baseline_lora_h16` 和 `pi05_libero_bsp_lora_h16`。LoRA 候选从 64 开始，
   按 64/32/16/8/4/2/1 递减并选择 A/B 共同稳定的最大值。
4. baseline 与 BSP 不得并行；每项都必须检查有限 metrics、目标 checkpoint、Orbax
   临时目录、GPU 释放和磁盘余量。单步成功不能替代 10/100-step 稳定性门禁。
5. 任一路线的两组 100-step pilot 通过后暂停，不自动启动 30k。正式训练前先复核
   checkpoint 容量和持久性：`/root` 未证明跨 DSW 重建持久，`ossfs2` 也未证明满足
   Orbax POSIX 语义。

自动监控 `0-5-libero-pilot` 已因 mb8 OOM 暂停；在用户批准新的显存方案前不得自动恢复。

## 13. 本批增量工作纪要（截至 2026-08-08）

### 13.1 GCS 下载工具与官方 checkpoint

1. `pi05_libero` 使用 `gcsfs` 低并发下载时进度长期停留在约 2%。没有并发启动第二个
   写同一 `.partial` 的进程，而是先停止原进程，再切换下载器。
2. 下载并核验 Google Cloud CLI 578.0.0 独立归档。最初给出的 SHA-256 期望值错误；
   通过归档精确大小、tar 完整性、内部版本以及与官方 GCS 响应一致的 MD5 排除了文件
   损坏。
3. 工具解压到项目专用 `tools/`，没有安装到系统目录；后续使用 `gsutil -m` 完成
   `pi05_libero`。
4. 最终目录存在 `params`，原 `.partial` 消失，`du -sh` 为 12 GiB。资产目录中存在
   `physical-intelligence/libero/norm_stats.json`。

### 13.2 资产身份固化

1. 记录官方 norm stats 的 SHA-256。
2. 下载并验证 PaliGemma tokenizer，记录精确字节数和 SHA-256。
3. 顺序读取两个约 12 GiB checkpoint，分别生成包含文件路径和内容的目录树摘要；
   两项格式门禁均通过。
4. 这些摘要用于证明 A/B 训练的共同基座和官方环境校准模型未变化，不冒充上游发布的
   官方整目录 checksum。

### 13.3 双 Python 环境进度

1. OpenPI 继续使用 CPython 3.11.9 和既有 `/venvs/openpi`。
2. uv 另行安装 CPython 3.8.20，并创建 `/venvs/libero-py38`。
3. `sys.prefix`、`sys.base_prefix` 和 Python 3.8 版本门禁通过。uv 对仓库根项目
   `requires-python >=3.11` 的提示只反映 workspace 元数据，不表示显式 3.8 venv 创建失败。
4. 本小节记录的是创建 venv 当时的阶段性状态；后续依赖同步、editable 安装、freeze、
   私有 EGL vendor、真实 reset/render/step 和官方 rollout 均已完成，见第 6、7、14 节。

### 13.4 未发生的变更

- 没有配置命令行代理，也没有确认阿里云 GA、NAT Gateway、EIP 或 SNAT。
- 没有修改系统 Python 3.12、系统 CUDA、NVIDIA 驱动、Docker 或 Container Toolkit。
- 没有向 `/mnt/data` 下除 `/mnt/data/siyuanxue` 以外的目录写入。
- 截至该阶段性批次当时还没有构建 sidecar、norm、训练或仿真；这些状态已被后续第 14 节
  更新覆盖，不能再用本条判断当前服务器进度。

## 14. 后续工作纪要（截至 2026-08-08 23:03，Asia/Shanghai）

### 14.1 LIBERO 隔离环境与 EGL

1. 仿真环境完成锁定依赖安装，并生成 `libero-py38-freeze.txt`。输入身份包括：
   `uv.lock=4be03e65...`、OpenPI LIBERO requirements `821008c9...`、LIBERO 子模块
   requirements `23ef4abd...`、openpi-client pyproject `cacd5539...`；freeze 文件摘要为
   `7b022200...`。
2. DSW 容器只暴露 `NVIDIA_DRIVER_CAPABILITIES=compute,utility`，没有 `/dev/dri`，系统
   GLVND vendor 目录也不存在，但 NVIDIA GPU 设备节点、`libEGL_nvidia.so.0` 和依赖完整。
3. 项目私有 `cache/egl/10_nvidia.json` 的 SHA-256 为 `c4ddd234...`；通过进程级环境变量
   使用，不修改 `/etc` 或 `/usr/share`。MuJoCo EGL context 与 LIBERO 真实画面均通过。
4. runtime identity 文件合成摘要为
   `sha256:81eba13e9882cd1ef02867b3e192d3b347203949ecae35ef55884ced1f662289`；该值是当前
   isolated-host 环境身份，不是 Docker image digest。

### 14.2 官方 checkpoint 仿真校准

1. `pi05_libero` policy server 成功恢复 checkpoint、加载官方 norm stats，并在本机
   8000 端口通过 health check。推理时 GPU 进程曾占用约 73,304 MiB。
2. task 0 单回合 smoke 成功后，顺序完成四套件 200 回合，196 次成功，宏平均 0.98；
   所有 episode 都有审计记录，没有 infrastructure incomplete 或 artifact error。
3. 校准目录包含 `episodes.jsonl`、`tasks.csv`、`suites.csv`、`summary.json`、manifest
   和选择性视频。目录约 3.9 MiB、44 个 MP4；policy server 随后正常停止。
4. 曾为校准结果规划
   `/mnt/data/siyuanxue/openpi-bsp/experiments/eval-archives/`，并通过目标路径无碰撞门禁；
   现有对话没有最终 tar 写入/校验输出，因此本文件不把该评测归档写成“已持久化完成”。

### 14.3 BSP sidecar、verification 与 norm

1. sidecar build 完成：259,121 个唯一 spline target 映射全部 273,465 frames；结构、dtype、
   有限值、mapping 范围和 manifest 均通过。
2. 当前 fingerprint 连续两次重算一致，完整 verify 通过；sidecar 与 verification 已分别
   写入 `/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/`。
3. baseline 与 BSP norm 各自完成 1,068-batch 扫描；state stats 完全一致，action stats
   明确隔离。norm tar 首次因未恢复 `ASSETS_BASE` 失败，修复变量并重建后用户确认全部门禁通过。

### 14.4 显存策略决策与单步探针

1. 既定模型仍为 π0.5、action horizon 16、action dim 32、全量微调、有效 batch 256、
   seed 42；梯度累积保证 micro-batch 变化不改变 optimizer-step 的有效 batch。
2. 保留 EMA 的全量微调在 micro-batch 1 也失败：`XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`
   时曾因额外 288 MiB 分配 OOM；提高到 0.95 的另一次尝试仍因约 2.73 GiB 分配 OOM。
3. 用户最终选择先执行“无 EMA 的全量微调”，而不是 LoRA。此决定降低显存，但会改变
   原计划中的 EMA 权重平滑协议，后续报告必须显式披露，不能称为与 EMA 完全等价。
4. 无 EMA、fraction 0.95 的 baseline/BSP micro-batch 1 单步探针都成功保存 checkpoint；
   每个 checkpoint 约 30–31 GiB。随后 micro-batch 2、4 的双组单步门禁也通过；旧 mb1/mb2
   大 checkpoint 经路径、非软链、params 和日志检查后清理，mb4 保留。
5. baseline/BSP micro-batch 64 的单步探针也都通过，checkpoint 约 29/31 GiB，但这只能
   证明一个 optimizer step，不能证明持续训练稳定。

### 14.5 持续训练失败证据

1. baseline micro-batch 64 的 100-step pilot 在 `pmicrobatch_grad` 处 OOM，尝试额外分配
   `15,186,328,288` bytes（约 14.14 GiB）；没有合法 100-step checkpoint，BSP mb64
   pilot 没有启动。
2. 随后按共同参数启动 baseline micro-batch 8 的 10-step pilot。Step 1 成功记录：
   `loss=0.0720`、`grad_norm=0.3417`、`param_norm=1802.3865`；下一步在
   `pmicrobatch_grad` 申请 `3,970,432,328` bytes（约 3.70 GiB）时
   `RESOURCE_EXHAUSTED`。
3. mb8 失败后进程已退出、GPU 空闲，目标 `10/params` 不存在，未发现 Orbax 临时目录；
   当时 `/root` 约 173 GiB 可用，因此直接失败原因是 GPU OOM，而不是磁盘耗尽。
4. 串行自动化遵守门禁：没有启动 BSP mb8、没有重试或降参、没有删除 checkpoint，
   并已暂停。单步探针通过与连续 10/100-step 稳定是两个不同门禁。

### 14.6 操作与审计边界

- 后期在用户明确授权后，助手使用 Browser/Computer Use 操作已登录的阿里云 Web VS Code
  终端，读取最新 UI 后输入命令；没有获得独立 SSH 凭据，也没有向公开文档写入主机名、
  Pod ID、GPU UUID、token 或 credential。
- 自动化只允许串行启动一项 GPU 作业；每阶段要求无 GPU compute process、路径无碰撞、
  `/root` 至少 80 GiB 可用，并在异常时暂停。
- 本轮文档整理只修改本地仓库中的本 Markdown，不对服务器执行命令，也不提交或推送。

## 15. 2026-08-09 LoRA 扩展决策

1. 现有 `pi05_libero_baseline_h16` 和 `pi05_libero_bsp_h16` 保持不变；LoRA 不是覆盖或
   隐式降级，而是两个独立配置：`pi05_libero_baseline_lora_h16` 和
   `pi05_libero_bsp_lora_h16`。
2. 两个 LoRA 配置均使用 OpenPI 官方 JAX LoRA 变体：PaliGemma `gemma_2b_lora`、action
   expert `gemma_300m_lora`，并使用与该模型配置严格匹配的 `get_freeze_filter()`。
3. LoRA 仍沿用第一阶段的 π0.5、horizon 16、action dim 32、LIBERO v2.0、seed 42、
   effective batch 256、阶段一学习率/AdamW、30k 和固定 checkpoint 周期。它只改变
   可训练参数集合。
4. `ema_decay=None` 与 OpenPI 官方 LoRA 示例一致，不把全量模型 EMA 副本重新引入显存。
   全量路线当前也由用户选择无 EMA，但两条路线仍必须在 manifest 中分别标注训练配置。
5. LoRA 不改变数据分布或 BSP 表示，因此复用已经验收的 baseline/BSP norm stats；在
   新配置的 assets 目录中原子复制后，目标 SHA-256 必须与来源完全一致，不重新扫描数据。
6. LoRA 代码无论全量路线是否成功都会进入仓库；只有全量 micro-batch 2 的持续训练门禁
   失败时才启动 LoRA GPU pilot。任一路线通过 A/B 100-step 后都暂停，等待正式训练复核。

## 16. 2026-08-09 固定验收扩展：0k 与 5k

用户已确认将第一阶段固定里程碑从 `10k/20k/30k` 扩展为
`0k/5k/10k/20k/30k`。本节记录协议决定和代码能力，不把尚未发生的服务器产物写成已完成：

1. `0k` 的严格含义是对应 Baseline 或 BSP 配置已经从同一个 `pi05_base` 加载权重，但
   optimizer update 次数仍为 0；它不是官方 `pi05_libero` checkpoint。
2. 全量和 LoRA 四个第一阶段配置均声明
   `permanent_checkpoint_steps=(0, 5000, 10000, 20000, 30000)`；每 1,000 step 的恢复保存与
   `keep_period=10000` 保持不变，但永久保留集合不会额外包含 15k/25k。
3. step 0 必须走正常 Orbax 保存路径，包含 `params/`、`train_state/` 和对应配置的 norm
   assets；只有 `0/` 时执行 `--resume` 必须从 step 0 恢复，并从 step 1 继续。
4. 固定比较现在要求同一训练家族的十个 run：Baseline/BSP 各五个里程碑。每个 run 仍为
   2,000 episodes，因此完整验收为 20,000 episodes；全量与 LoRA run 禁止混入同一报告。
5. 报告仍只生成六种审计文件，但学习曲线固定显示五个点，不选择 best checkpoint。
6. 截至写入本节时，上述改动仍需完成整套测试、合并到 `main`、推送并由服务器
   `pull --ff-only` 后才会生效；服务器上尚未产生正式 0k/5k checkpoint，也未开始正式
   30k 训练。

## 17. 2026-08-09 实施与服务器门禁增量

本节按终端已观察证据记录；未取得终态的项目明确标为未确认：

1. Baseline LoRA、micro-batch 64、无 EMA 的 100-step pilot 已正常退出；10–100 step 的
   十组 `loss/grad_norm/param_norm` 均为有限值，`100/params` 与 `100/train_state` 存在，
   日志无 OOM/RESOURCE_EXHAUSTED/Traceback，未发现 Orbax 临时目录。
2. BSP LoRA 100-step 的首次启动命令因手工抄写的完整 code SHA 不匹配而在前置门禁停止，
   没有创建日志、PID 或 checkpoint。改用服务器实际 SHA 后进程进入启动，但在训练状态和
   数据初始化前因新 shell 没有显式设置 `WANDB_MODE=offline` 而退出，日志报
   `api_key not configured (no-tty)`。这不是 OOM，也不能作为 BSP 稳定性结论。
3. 按异常停止规则，没有删除上述失败日志，也没有自动用同名或新名称重试 BSP pilot；
   下一次合法启动必须使用唯一实验名并在进程环境中显式传入 `WANDB_MODE=offline`。
4. 为 0k/5k 改动创建了本地功能分支 `feat/phase1-zero-five-k`；服务器主工作区仍停留在
   `b8f88bb...`，另建 detached 隔离 worktree 读取功能分支，不修改服务器 `main`。
5. 服务器锁定的 JAX/Orbax 环境对两个新 step-0 集成测试给出预期 RED：两项均失败，
   `2 failed, 2 deselected`，耗时约 75 秒；GPU 为空。失败分支日志保存在项目 logs 目录。
6. 随后已推送 step-0 生产实现，但重新登录后的终端证据显示隔离 worktree 仍停在
   `008196e`，预定的 GREEN 日志不存在；因此同两项 GREEN 测试实际**未启动**，不能视为
   测试中断或测试通过。在隔离 worktree 快进到 `ad01abe` 并取得明确 PASS 终态前，不得据此
   合并 `main` 或开始正式训练。

## 18. 2026-08-09 第一阶段缩短为 10k

由于单组 30k 训练按实测速度预计约需十天，用户批准用预先固定的短周期协议替代原 30k
协议。这个决定改变训练预算和验收横轴，但不改变模型、数据、优化器、有效 batch、seed、
Baseline/BSP 配对方式或每个 checkpoint 的完整评测规模。

1. 被替代的 Baseline LoRA 正式运行使用代码
   `196651804f21d25f4b92f0f0d67801e42b140089`、配置
   `pi05_libero_baseline_lora_h16` 和实验名 `phase1-seed42-baseline`。身份核对后向 PID
   `1718659` 发送 `SIGTERM`，进程正常退出；最后观察到 `212 / 30,000`，日志异常匹配为 0，
   GPU compute process 为 0，旧运行只存在 step 0。旧日志和 checkpoint 全部保留，不再恢复
   或覆盖该实验。
2. 终止审计文件为
   `/root/openpi-bsp-work/experiments/logs/protocol-transition-30k-to-10k-20260809T120259Z-2211334.txt`，
   SHA-256 为
   `36f9ba560adbad8caf946ff30f30a60a7ef57d4a24864ff2820d22e69d7aa7a6`。
3. 新协议的训练终点是 10,000 optimizer steps，固定永久里程碑是
   `0k/1k/2k/5k/10k`。每 1,000 step 保存恢复点，`keep_period=10000`，四个 phase-one
   full/LoRA Baseline/BSP 配置使用同一个里程碑集合。
4. 新正式实验名固定为 `phase1-short10k-seed42-baseline` 和
   `phase1-short10k-seed42-bsp`。它们必须从同一个 `pi05_base` 独立初始化，不能从旧 212-step
   运行恢复。
5. A/B 各评测五个 checkpoint，每次仍为四套件 × 10 tasks × 50 initial states = 2,000
   episodes；比较器仍要求恰好十个 run、20,000 episodes，并生成原来的六种固定报告文件。
6. 本节写入代码仓库时，新 10k 训练尚未启动。只有短周期配置、报告和 runbook 合同测试通过，
   功能分支合并并推送、服务器 `main` 快进到最终 SHA 且启动前门禁通过后，才允许用新实验名
   启动 Baseline。
