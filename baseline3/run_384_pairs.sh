#!/usr/bin/env bash
# =============================================================================
# 8 小时窗口：造「384 交付版」——给 T4 超时留一条退路
#
# 背景
# ----
# 双模型在 448 推理：5090 上 39ms → T4 折算约 100ms，**正压线**。
# 384 推理的前向只要 0.735 倍 → 约 90ms，能留出 10% 余量。
# 但 448 训练的权重直接降到 384 推理会掉 0.0026（实测 0.8027→0.8001），
# 所以这里**直接用 384 训练**，保证训练/推理同尺寸。
#
# 同时留一条「长训练」线：swin_b 在 30 轮里第 26 轮才见顶，说明它没训完。
#
# 车道：
#   A: swin_b 45 轮（长训练，测它到底能到多少）
#   B: in22k_384（12 轮，22k 起点 + 384）→ effv2m_384（12 轮）
#   C: effv2s_384（12 轮）→ cnx_small_384（12 轮）
#
# 为什么 12 轮：12 轮那批（0.8066）比 30 轮那批（0.8011~0.8061）不差，且省 60% 时间。
#
# 用法：
#   cd ~/cxr/baseline3
#   tmux new -s p384
#   bash run_384_pairs.sh 2>&1 | tee runs/p384.log
# =============================================================================
set -u
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-12}"
BS="${BS:-12}"
WORKERS="${WORKERS:-6}"
BUDGET_MIN="${BUDGET_MIN:-460}"
STAMP=$(date +%m%d_%H%M)
LOG="runs/p384_$STAMP"
mkdir -p "$LOG"
START=$SECONDS

say() { echo "[$(date +%H:%M:%S)] $*"; }
remain() { echo $(( BUDGET_MIN - (SECONDS - START) / 60 )); }

SSL="../baseline1/runs/ssl/simclr200_384.pth"
SW22K="weights/convnext_base_22k_1k_224.pth"

GPUS=()
while IFS= read -r g; do [ -n "$g" ] && GPUS+=("$g"); done < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, '$2 < 1500 {print $1}' | head -3
)
if [ "${#GPUS[@]}" -eq 0 ]; then say "✗ 没有空闲 GPU"; exit 1; fi
say "可用 GPU：${GPUS[*]}｜预算 ${BUDGET_MIN} 分钟｜epochs ${EPOCHS}"

HAS22K=0
[ -f "$SW22K" ] && [ "$(stat -c%s "$SW22K" 2>/dev/null || echo 0)" -gt 300000000 ] && HAS22K=1

lane() {
  local gpu="$1"; shift
  for spec in "$@"; do
    local name="${spec%%|*}"; local rest="${spec#*|}"
    local est="${rest%%|*}"; local cmd="${rest#*|}"
    if [ -f "runs/$name/best.pth" ]; then say "  [G$gpu] 跳过 $name（已存在）"; continue; fi
    if [ "$(remain)" -lt "$est" ]; then say "  [G$gpu] 跳过 $name（剩余 $(remain) < ${est}）"; continue; fi
    say "  [G$gpu] ▶ $name（预计 ${est} 分钟）"
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash -c "$cmd" > "$LOG/$name.log" 2>&1
    local rc=$? used=$(( (SECONDS - t0) / 60 ))
    [ "$rc" -eq 0 ] && say "  [G$gpu] ✓ $name（${used} 分钟）" \
                    || say "  [G$gpu] ✗ $name（码 $rc，${used} 分钟）见 $LOG/$name.log"
  done
}

A=("swin_b45|300|python -u train3.py --arch swin_b --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs 45 --seed 48 --aug mild --out runs/swin_b45")
B=()
if [ "$HAS22K" -eq 1 ]; then
  B+=("in22k_384|110|python -u train3.py --arch convnext_base_in22k --pretrained $SW22K --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/in22k_384")
fi
B+=("effv2m_384|110|python -u train3.py --arch efficientnet_v2_m --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/effv2m_384")
C=("effv2s_384|90|python -u train3.py --arch efficientnet_v2_s --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/effv2s_384")
C+=("cnx_small_384|120|python -u train3.py --arch convnext_small --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/cnx_small_384")

say "【阶段 1】三车道启动"
PIDS=()
N=$(( ${#GPUS[@]} < 3 ? ${#GPUS[@]} : 3 ))
[ "$N" -ge 1 ] && { lane "${GPUS[0]}" "${A[@]}" & PIDS+=("$!"); }
[ "$N" -ge 2 ] && { lane "${GPUS[1]}" "${B[@]}" & PIDS+=("$!"); }
[ "$N" -ge 3 ] && { lane "${GPUS[2]}" "${C[@]}" & PIDS+=("$!"); }
for p in "${PIDS[@]}"; do wait "$p"; done
say "【阶段 1】结束（$(( (SECONDS - START) / 60 )) 分钟）"

say "【阶段 2】valid / test 预测"
G0="${GPUS[0]}"
for name in swin_b45 in22k_384 effv2m_384 effv2s_384 cnx_small_384; do
  [ -f "runs/$name/best.pth" ] || continue
  [ -f "runs/$name/pv.npz" ] || CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split valid \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pv.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pt.npz" ] || CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split test \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pt.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pt.npz" ] && say "  ✓ $name" || say "  ✗ $name"
done

say "【阶段 3】峰值汇总"
python - "$LOG" <<'PY' | tee "$LOG/summary.txt"
import json, glob, sys
rows = []
for d in sorted(glob.glob("runs/*/metrics.json")):
    n = d.split("/")[1]
    if not any(k in n for k in ("swin_b45", "_384")):
        continue
    try:
        m = json.load(open(d))
        if m:
            b = max(m, key=lambda x: x["study_auc"])
            rows.append((n, b["study_auc"], b["epoch"]))
    except Exception:
        pass
print(f"{'实验':<18}{'峰值':>8}{'轮':>5}")
print("-" * 34)
for n, a, e in sorted(rows, key=lambda x: -x[1]):
    print(f"{n:<18}{a:>8.4f}{e:>5}")
print("\n384 版的意义：推理耗时约为 448 的 0.735 倍，T4 折算从 ~100ms 降到 ~90ms。")
print("若 T4 实测 448 超标，就改用 384 版交付（模型/提交都要换，且要重新提交）。")
PY

echo
say "结束（累计 $(( (SECONDS - START) / 60 )) 分钟）｜看 $LOG/summary.txt"
echo "=============================================================="
