#!/usr/bin/env bash
# =============================================================================
# 凌晨挂机：补两个"不一样"的集成成员 + 赌一把 large@448
#
# 为什么是这三个任务（按顺序，时间不够就自动跳过后面的）：
#   1) res50_full     —— 你点名的：纯残差卷积，与 ConvNeXt/EfficientNetV2 归纳偏置不同。
#                        单用约 0.78~0.79，价值全在"当第三票"。45MB，几乎不占体积。
#   2) large448_full  —— 单模型上限最高的一条：large@384 已经 0.8059（平台 77.37），
#                        448 可能更高，而且**单模型不受多模型耗时叠加的拖累**。
#   3) swin_t_full    —— 窗口注意力，与卷积系差异最大。53MB。只在 1、2 都跑完才轮到。
#
# 为什么不做同架构多 seed：已实测 +0.0021（5 折）/ +0.0004（soup），错误相关性太高。
#
# 用法（服务器）：
#   cd ~/cxr/baseline1
#   tmux new -s night
#   bash run_night_members.sh 2>&1 | tee runs/night_members.log
#   # Ctrl-B 再 D
#
# 可调：BUDGET_MIN=300  BS=12  WORKERS=8
# =============================================================================
set -u
cd "$(dirname "$0")"

BUDGET_MIN="${BUDGET_MIN:-300}"
BS="${BS:-12}"
WORKERS="${WORKERS:-8}"
STAMP=$(date +%m%d_%H%M)
LOG="runs/night_$STAMP"
mkdir -p "$LOG"
START=$SECONDS

say() { echo "[$(date +%H:%M:%S)] $*"; }
remain() { echo $(( BUDGET_MIN - (SECONDS - START) / 60 )); }

# ---------- 选卡：空闲显存最多且 >= 要求值 ----------
pick_gpu() {          # $1 = 需要的最小空闲 MiB
  local need="${1:-9000}"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, -v n="$need" '$2 >= n {print $2" "$1}' \
    | sort -rn | head -1 | cut -d' ' -f2
}

wait_gpu() {          # 最多等 20 分钟
  local need="$1" waited=0 g
  while [ $waited -lt 1200 ]; do
    g=$(pick_gpu "$need")
    if [ -n "$g" ]; then echo "$g"; return 0; fi
    say "  暂无 >=$((need/1024))GB 空闲的卡，等 2 分钟（已等 $((waited/60)) 分钟）"
    sleep 120; waited=$((waited + 120))
  done
  echo ""
}

# ---------- run_job <名字> <预计分钟> <最小空闲MiB> <命令...> ----------
run_job() {
  local name="$1" est="$2" need="$3"; shift 3
  local out="runs/$name"

  if [ -f "$out/best.pth" ]; then
    say "跳过 $name：已有 best.pth（断点续跑）"
    return 0
  fi
  if [ "$(remain)" -lt "$est" ]; then
    say "跳过 $name：剩余 $(remain) 分钟 < 预计 ${est} 分钟"
    return 1
  fi

  local gpu; gpu=$(wait_gpu "$need")
  if [ -z "$gpu" ]; then
    say "跳过 $name：等 20 分钟仍无 >=$((need/1024))GB 空闲的卡"
    return 1
  fi

  local free_mb; free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
  say "▶ $name ｜GPU$gpu（空闲 ${free_mb}MiB）｜batch $BS｜预计 ${est} 分钟｜剩余 $(remain) 分钟"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$@" 2>&1 | tee "$LOG/$name.log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -ne 0 ]; then
    if grep -qi "out of memory" "$LOG/$name.log"; then
      say "✗ $name 显存不足。重跑时把 batch 调小：BS=8 bash run_night_members.sh"
    else
      say "✗ $name 失败（退出码 $rc），见 $LOG/$name.log 末尾"
    fi
    return 1
  fi

  # 顺手清掉每轮的 EMA 快照（每份 45~380MB，留着没用，best.pth 才是要交的）
  find "$out" -name "ema_f*.pth" -delete 2>/dev/null
  say "✓ 完成 $name（累计 $(( (SECONDS - START) / 60 )) 分钟，剩余 $(remain) 分钟）"
  return 0
}

echo "=============================================================="
say "预算 ${BUDGET_MIN} 分钟 ｜ batch ${BS} ｜ 日志 $LOG"
echo "=============================================================="

# =============================================================================
# 任务 1：resnet50 全量（12 轮）—— 你点名的第三票
#   lr：baseline1 默认 3e-4 @ batch32，按线性缩放到 batch12 约 1.1e-4
#   不带 --ssl：SSL 权重是 ConvNeXt 的，键名/形状都对不上
# =============================================================================
run_job res50_full 70 9000 \
  python -u train.py --backbone resnet50 --epochs 12 --img-size 448 \
    --batch-size "$BS" --lr 1.1e-4 --workers "$WORKERS" --seed 48 \
    --out runs/res50_full

# =============================================================================
# 任务 2：convnext_large @448 全量 —— 单模型上限最高，且没有多模型耗时叠加
#   batch 8（large 在 448 下显存需求大，必须 >=13GB 空闲才开）
#   lr 8e-5（batch8 线性缩放；large 对 LR 更敏感，取略保守）
# =============================================================================
run_job large448_full 150 13000 \
  python -u train.py --backbone convnext_large --epochs 12 --img-size 448 \
    --batch-size 8 --lr 8e-5 --workers "$WORKERS" --seed 48 \
    --out runs/large448_full

# =============================================================================
# 任务 3：swin_t 全量 —— 窗口注意力，与卷积系差异最大（只有前两个跑完才轮到）
# =============================================================================
run_job swin_t_full 80 9000 \
  python -u train.py --backbone swin_t --epochs 12 --img-size 448 \
    --batch-size "$BS" --lr 1.1e-4 --workers "$WORKERS" --seed 48 \
    --out runs/swin_t_full

# =============================================================================
# 收尾：出 valid 预测 + 评估所有集成组合 + 打印体积/耗时账
# =============================================================================
say "【收尾】生成 valid 预测并评估集成组合"

for n in res50_full large448_full swin_t_full; do
  [ -f "runs/$n/best.pth" ] || continue
  [ -f "runs/$n/pv.npz" ] && continue
  python -u predict.py --split valid --ckpt "runs/$n/best.pth" --out "runs/$n/pv.npz" \
    >>"$LOG/predict.log" 2>&1 \
    && say "  ✓ $n valid 预测已出" || say "  ✗ $n 预测失败（见 $LOG/predict.log）"
done

python - "$LOG" <<'PY' | tee "$LOG/members_report.txt"
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from common import (aggregate_by_study, encode_labels, macro_auc,
                    per_class_auc, read_split_csv)

log = sys.argv[1]
rows = read_split_csv("valid")
y = encode_labels(rows)

# 成员：名字 -> (npz, fp16 MB, 前向 ms 估值@448)
CAND = [
    ("convnext_base", "runs/full_base/pv2.npz",   169, 7.2),
    ("effv2m",        "runs/full_effv2m/pv.npz",  108, 14.0),
    ("resnet50",      "runs/res50_full/pv.npz",    45, 6.0),
    ("swin_t",        "runs/swin_t_full/pv.npz",   53, 8.0),
    ("large448",      "runs/large448_full/pv.npz",377, 21.0),
]
avail = [(n, p, mb, ms) for n, p, mb, ms in CAND if os.path.exists(p)]

def load(p):
    return np.load(p, allow_pickle=True)["probs"].astype(np.float64)

def score(ps):
    m = np.mean(ps, axis=0)
    _, sp, sl = aggregate_by_study(rows, m, how="mean", labels=y)
    return macro_auc(sl, sp), per_class_auc(sl, sp)

print(f"可用成员：{[a[0] for a in avail]}\n")
print(f"{'组合':<42}{'valid':>8}{'体积MB':>8}{'前向ms':>8}")
print("-" * 68)

results = {}
combos = [
    ("convnext_base（单模型）",            ["convnext_base"]),
    ("convnext + effv2m（现役 77.67）",    ["convnext_base", "effv2m"]),
    ("convnext + resnet50",                ["convnext_base", "resnet50"]),
    ("convnext + effv2m + resnet50",       ["convnext_base", "effv2m", "resnet50"]),
    ("convnext + effv2m + swin_t",         ["convnext_base", "effv2m", "swin_t"]),
    ("convnext + effv2m + resnet50 + swin_t", ["convnext_base", "effv2m", "resnet50", "swin_t"]),
    ("large448（单模型）",                  ["large448"]),
    ("large448 + convnext",                ["large448", "convnext_base"]),
]
lut = {n: (p, mb, ms) for n, p, mb, ms in avail}
for label, members in combos:
    if not all(m in lut for m in members):
        print(f"{label:<42}{'(缺成员)':>8}")
        continue
    ps = [load(lut[m][0]) for m in members]
    auc, cls = score(ps)
    size = sum(lut[m][1] for m in members)
    fwd = sum(lut[m][2] for m in members)
    flag = "" if size <= 500 else "  ✗超体积"
    print(f"{label:<42}{auc:>8.4f}{size:>8}{fwd:>8.1f}{flag}")
    results[label] = auc

print("\n【判据】")
print("  · 三/四模型组合只有在体积 ≤500MB **且** T4 实测 ≤100ms 时才可用；")
print("  · 体积按 fp16 估，前向 ms 是 5090@448 的粗略值，T4 折算约 ×3.8；")
print("  · 只比 valid 的话，0.002 以内都当噪声（859 study，±0.01）。")
print(f"\n日志目录：{log}")
PY

echo
say "结束。剩余预算 $(remain) 分钟。"
say "看这个文件：$LOG/members_report.txt"
echo "=============================================================="
