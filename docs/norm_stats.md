# LIBERO phase-one normalization statistics

第一阶段必须为 baseline 和 BSP 分别计算 action normalization statistics，同时证明两组看到
完全相同的 state distribution。不能复用官方 `pi05_libero` 的统计，也不能让一组读取另一组
的 action stats。

## 为什么必须分离

`scripts/compute_norm_stats.py` 在 model padding 前统计：

```text
baseline actions: [16, 7]   7D native LIBERO chunks
BSP actions:      [16, 8]   controls[0:7] + knot[7]
state:            [8]       both variants use the same observations
```

baseline 和 BSP 的 action 语义、维度和分布不同，共用 stats 会把 spline knot 当成普通动作，
或让模型按错误区间缩放。相反，state 的 `mean/std/q01/q99` 应在给定容差内逐项相同；不相同
说明数据选择、episode mapping 或 transforms 已经漂移。

## 规范路径

```bash
export BSP_WORK=/root/openpi-bsp-work
export BSP_REPO_DIR="$BSP_WORK/repo/openpi05-bsp"
export OPENPI_PY="$BSP_WORK/venvs/openpi/bin/python"
export ASSETS_BASE="$BSP_WORK/experiments/assets"
export LIBERO_DATASET_DIR=/mnt/data/siyuanxue/openpi-bsp/data/lerobot/physical-intelligence/libero
export BSP_CACHE=/mnt/data/siyuanxue/openpi-bsp/data/bspline-targets/libero-v2.0-bsp-v2.npz

export BASELINE_FULL_ASSETS="$ASSETS_BASE/pi05_libero_baseline_h16"
export BSP_FULL_ASSETS="$ASSETS_BASE/pi05_libero_bsp_h16"
export BASELINE_FULL_NORM="$BASELINE_FULL_ASSETS/libero_baseline_h16"
export BSP_FULL_NORM="$BSP_FULL_ASSETS/libero_bsp_h16"

export BASELINE_LORA_NORM="$ASSETS_BASE/pi05_libero_baseline_lora_h16/libero_baseline_h16"
export BSP_LORA_NORM="$ASSETS_BASE/pi05_libero_bsp_lora_h16/libero_bsp_h16"
export NORM_COMPARISON="$ASSETS_BASE/libero-phase1-norm-comparison.json"
```

所有数据盘写路径都位于 `/mnt/data/siyuanxue`。stats 本身是小文件，活跃训练从快速本地
`/root` 读取；通过后再按服务器 runbook 归档。

## 计算 baseline 与 BSP

先 baseline，再 BSP；不要同时跑两个完整 DataLoader：

```bash
cd "$BSP_REPO_DIR"
test ! -e "$BASELINE_FULL_NORM/norm_stats.json"
test ! -e "$BSP_FULL_NORM/norm_stats.json"
test ! -e "$NORM_COMPARISON"

"$OPENPI_PY" scripts/compute_norm_stats.py \
  pi05_libero_baseline_h16 \
  --assets-dir "$BASELINE_FULL_ASSETS" \
  --dataset-root "$LIBERO_DATASET_DIR"

"$OPENPI_PY" scripts/compute_norm_stats.py \
  pi05_libero_bsp_h16 \
  --assets-dir "$BSP_FULL_ASSETS" \
  --dataset-root "$LIBERO_DATASET_DIR" \
  --bsp-cache-path "$BSP_CACHE" \
  --compare-state-stats-with "$BASELINE_FULL_NORM" \
  --norm-comparison-output "$NORM_COMPARISON"
```

`max_frames` 只允许用于开发测试，不允许生成正式 stats。官方 LIBERO v2.0 的 warning 不要求
执行 v2.1 转换。

## 必须通过的 comparison gate

```bash
"$OPENPI_PY" - "$NORM_COMPARISON" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("state_stats_equal", "asset_directories_isolated", "action_stats_isolated"):
    assert result[key] is True, (key, result[key])
for field in ("mean", "std", "q01", "q99"):
    row = result["state_fields"][field]
    assert row["equal"] is True, (field, row)
assert result["baseline_action_stats_sha256"] != result["bsp_action_stats_sha256"]
print("phase1_norm_comparison_gate=PASS")
PY
```

此外必须确认：

- 两个文件的 state 和 actions `mean/std/q01/q99` 全部有限；
- state shape 为 `[8]`；
- baseline action shape 为 `[7]`，BSP action shape 为 `[8]`；
- BSP knot channel 是 index 7，`q01 < q99`，没有退化区间；
- 两个 asset 目录不同，两个完整文件 SHA256 被记录。

`scripts.compute_norm_stats._validate_bsp_action_stats` 和 `_stats_sha256` 是代码门禁；不要手工
编辑 JSON 来绕过失败。comparison 中 state 使用 `rtol=1e-7`、`atol=1e-8`，不是字符串比较。

## 发布给 LoRA 配置

full 与 LoRA 配置的 action/state protocol 相同，只是可训练参数不同，因此 LoRA 复用已经通过
门禁的同一 variant stats。配置名决定 asset 根目录，必须显式发布到 LoRA 路径；仅留下 full
目录会导致训练找不到 stats。

以下操作先在目标目录写临时文件，再在同一目录原子 rename；目标存在时停止，不覆盖：

```bash
mkdir -p "$BASELINE_LORA_NORM" "$BSP_LORA_NORM"
test ! -e "$BASELINE_LORA_NORM/norm_stats.json"
test ! -e "$BSP_LORA_NORM/norm_stats.json"
test ! -e "$BASELINE_LORA_NORM/.norm_stats.json.publish-tmp"
test ! -e "$BSP_LORA_NORM/.norm_stats.json.publish-tmp"

cp "$BASELINE_FULL_NORM/norm_stats.json" \
  "$BASELINE_LORA_NORM/.norm_stats.json.publish-tmp"
mv "$BASELINE_LORA_NORM/.norm_stats.json.publish-tmp" \
  "$BASELINE_LORA_NORM/norm_stats.json"

cp "$BSP_FULL_NORM/norm_stats.json" \
  "$BSP_LORA_NORM/.norm_stats.json.publish-tmp"
mv "$BSP_LORA_NORM/.norm_stats.json.publish-tmp" \
  "$BSP_LORA_NORM/norm_stats.json"

test "$(sha256sum "$BASELINE_FULL_NORM/norm_stats.json" | awk '{print $1}')" = \
  "$(sha256sum "$BASELINE_LORA_NORM/norm_stats.json" | awk '{print $1}')"
test "$(sha256sum "$BSP_FULL_NORM/norm_stats.json" | awk '{print $1}')" = \
  "$(sha256sum "$BSP_LORA_NORM/norm_stats.json" | awk '{print $1}')"
```

发布前若已有文件，先创建带时间戳的备份目录并记录 SHA；不要静默覆盖。

## 训练与评测绑定

每个 checkpoint 的 `assets/<asset_id>/norm_stats.json` 必须与对应训练前文件逐字节同 hash：

```text
baseline -> libero_baseline_h16 -> baseline norm hash
BSP      -> libero_bsp_h16      -> BSP norm hash
```

每个 evaluator manifest 记录 checkpoint 内实际文件的 SHA256。十个 run 中，同一 variant 的
norm hash 必须保持一致；baseline 与 BSP 必须不同。最终 reporter 会再次读取原始
`libero-phase1-norm-comparison.json`，拒绝 state gate 失败、action hash 相同、路径混用或
manifest hash 漂移。

本地 CPU 合同只能验证代码规则；完整 273,465-frame stats 计算与实际 JSON 数值仍是服务器
门禁，不能声称瘦身分支已经运行过。
