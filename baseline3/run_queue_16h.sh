#!/usr/bin/env bash
# =============================================================================
# 16 小时任务队列（baseline3）—— 三卡三车道，跑完 6 个成员
#
# 为什么这样排
# -----------
# 五个提交点已证明 **valid 排序对平台没有预测力**（0.8059→77.37，0.8027→77.67）。
# 所以这批任务的目标不是"把 valid 刷高"，而是**产出多个彼此不同、且都能交付的候选**：
#   · 每个成员单独 ≤170MB，两两组合必在 500MB 内；
#   · 任意候选只取 2 个成员（双模型已 39ms，T4 约 100ms，没有余量再加）。
#
# 车道设计（每车道串行，三车道并行）：
#   A: cnx30_strong（30轮 强增强+Mixup，带 SSL）  →  in22k30（ImageNet-22k 起点，唯一没验过的起点）
#   B: eff30_strong（30轮 同配方）                →  effv2s30（41MB，最轻的成员）
#   C: cnx_small30（30轮 常规增强）               →  swin_b30（167MB，与卷积系差异最大）
#
# 为什么是 30 轮：我们之前全用 12 轮，而强增强配 12 轮是欠训练（12 轮时 ep7~10 就到峰值）。
# 文献里 CXR 的标准配方是 30~50 轮 + 重增强，这个组合我们从没试过。
#
# 用法：
#   cd ~/cxr/baseline3
#   tmux new -s q16
#   bash run_queue_16h.sh 2>&1 | tee runs/q16.log
#
# 可调：BUDGET_MIN=460（一个晚上）｜BS=12｜WORKERS=6｜EPOCHS=30
# 断点续跑：直接重跑同一条命令，已有 best.pth 的任务会跳过。
# =============================================================================
set -u
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-30}"
BS="${BS:-12}"
WORKERS="${WORKERS:-6}"
BUDGET_MIN="${BUDGET_MIN:-460}"
STAMP=$(date +%m%d_%H%M)
LOG="runs/q16_$STAMP"
mkdir -p "$LOG"
START=$SECONDS

say()  { echo "[$(date +%H:%M:%S)] $*"; }
remain() { echo $(( BUDGET_MIN - (SECONDS - START) / 60 )); }

SSL="../baseline1/runs/ssl/simclr200_384.pth"
SW22K="weights/convnext_base_22k_1k_224.pth"

# ---------- 空闲卡 ----------
GPUS=()
while IFS= read -r g; do [ -n "$g" ] && GPUS+=("$g"); done < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, '$2 < 1500 {print $1}' | head -3
)
if [ "${#GPUS[@]}" -eq 0 ]; then
  say "✗ 没有空闲 GPU（<1.5GB 占用）。等别人跑完再重跑本脚本。"
  exit 1
fi
say "可用 GPU：${GPUS[*]}（${#GPUS[@]} 张）｜预算 ${BUDGET_MIN} 分钟｜epochs ${EPOCHS}｜batch ${BS}"

# ---------- 22k 权重检查 ----------
HAS22K=0
if [ -f "$SW22K" ] && [ "$(stat -c%s "$SW22K" 2>/dev/null || echo 0)" -gt 300000000 ]; then
  HAS22K=1; say "IN-22k 权重 ✓（$(du -h "$SW22K" | cut -f1)）"
else
  say "⚠ IN-22k 权重缺失或体积不对（应约 338MB）→ 车道 A 第二棒改为跳过"
fi

# =============================================================================
# 车道函数：串行执行 "名字|预计分钟|命令"
# =============================================================================
lane() {
  local gpu="$1"; shift
  for spec in "$@"; do
    local name="${spec%%|*}"; local rest="${spec#*|}"
    local est="${rest%%|*}"; local cmd="${rest#*|}"

    if [ -f "runs/$name/best.pth" ]; then
      say "  [GPU$gpu] 跳过 ${name}（已存在 best.pth）"; continue
    fi
    if [ "$(remain)" -lt "$est" ]; then
      say "  [GPU$gpu] 跳过 ${name}（剩余 $(remain) 分钟 < 预计 ${est}）"; continue
    fi

    say "  [GPU$gpu] ▶ ${name}（预计 ${est} 分钟，剩余 $(remain)）"
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash -c "$cmd" > "$LOG/$name.log" 2>&1
    local rc=$?
    local used=$(( (SECONDS - t0) / 60 ))

    if [ "$rc" -eq 0 ]; then
      say "  [GPU$gpu] ✓ ${name}（实际 ${used} 分钟）"
    else
      say "  [GPU$gpu] ✗ $name 失败（码 ${rc}，${used} 分钟），日志 $LOG/$name.log"
      grep -qi "out of memory" "$LOG/$name.log" 2>/dev/null && say "      → OOM：下次用 BS=8 重跑"
    fi
  done
}

# =============================================================================
# 组车道任务
# =============================================================================
A_JOBS=("cnx30_strong|210|python -u train3.py --arch convnext_base --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug strong --mixup 0.2 --ssl $SSL --out runs/cnx30_strong")
B_JOBS=("eff30_strong|210|python -u train3.py --arch efficientnet_v2_m --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug strong --mixup 0.2 --out runs/eff30_strong")
C_JOBS=("cnx_small30|190|python -u train3.py --arch convnext_small --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/cnx_small30")

if [ "$HAS22K" -eq 1 ]; then
  A_JOBS+=("in22k30|210|python -u train3.py --arch convnext_base_in22k --pretrained $SW22K --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug strong --mixup 0.2 --out runs/in22k30")
fi
B_JOBS+=("effv2s30|170|python -u train3.py --arch efficientnet_v2_s --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/effv2s30")
C_JOBS+=("swin_b30|230|python -u train3.py --arch swin_b --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/swin_b30")

# =============================================================================
# 并行启动三车道
# =============================================================================
say "【阶段 1】启动车道"
N=$(( ${#GPUS[@]} < 3 ? ${#GPUS[@]} : 3 ))
PIDS=()
[ "$N" -ge 1 ] && { lane "${GPUS[0]}" "${A_JOBS[@]}" & PIDS+=("$!"); }
[ "$N" -ge 2 ] && { lane "${GPUS[1]}" "${B_JOBS[@]}" & PIDS+=("$!"); }
[ "$N" -ge 3 ] && { lane "${GPUS[2]}" "${C_JOBS[@]}" & PIDS+=("$!"); }

for p in "${PIDS[@]}"; do wait "$p"; done
say "【阶段 1】全部车道结束（累计 $(( (SECONDS - START) / 60 )) 分钟）"

# =============================================================================
# 阶段 2：出预测
# =============================================================================
say "【阶段 2】valid / test 预测"
G0="${GPUS[0]}"
for name in cnx30_strong eff30_strong cnx_small30 in22k30 effv2s30 swin_b30; do
  [ -f "runs/$name/best.pth" ] || continue
  [ -f "runs/$name/pv.npz" ] || \
    CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split valid \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pv.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pt.npz" ] || \
    CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split test \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pt.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pt.npz" ] && say "  ✓ $name" || say "  ✗ ${name}（见 $LOG/predict.log）"
done

# =============================================================================
# 阶段 3：峰值汇总（valid 只作记录，不作选型依据）
# =============================================================================
say "【阶段 3】峰值汇总"
python - "$LOG" <<'PY' | tee "$LOG/summary.txt"
import json, glob, os, sys
rows = []
for d in sorted(glob.glob("runs/*/metrics.json")):
    n = d.split("/")[1]
    if not any(k in n for k in ("30_strong", "small30", "in22k30", "effv2s30", "swin_b30")):
        continue
    try:
        m = json.load(open(d))
        if not m: continue
        b = max(m, key=lambda x: x["study_auc"])
        rows.append((n, b["study_auc"], b["epoch"]))
    except Exception:
        pass
print(f"{'实验':<16}{'峰值':>8}{'轮':>5}")
print("-" * 32)
for n, a, e in sorted(rows, key=lambda x: -x[1]):
    print(f"{n:<16}{a:>8.4f}{e:>5}")
print("\n注意：valid 不用于选型（5 个提交点证明无预测力）。")
print("下一步：挑 2 成员、体积<450MB 的组合，跑 cost 确认 0 张不一致，再用提交额度去试。")
PY

echo
say "结束（累计 $(( (SECONDS - START) / 60 )) 分钟，剩余预算 $(remain) 分钟）"
say "看：$LOG/summary.txt ｜ runs/*/best.pth ｜ $LOG/*.log"
echo "=============================================================="
