#!/usr/bin/env bash
# =============================================================================
# 接下来 8 小时（baseline3）：把 22k 起点做"落定版"，并给 T4 留 384 退路
#
# 依据（11-08 上午的实测）
# ----------------------
# 1) **ImageNet-22k 起点有效 +0.0050**（cnx30_strong 0.8011 vs in22k30 0.8061，
#    同架构/同强增强/同 30 轮，只换起点）——DINOv2 失败后第一次测到起点有收益。
# 2) **30 轮没帮上**：eff30_strong 0.8018 反而低于 12 轮那批 0.8066 → 12 轮性价比更高。
#    而且 in22k30 在 **ep7** 就见顶，30 轮是浪费。
# 3) **swin_b 在 30 轮的第 26 轮才见顶** → 它没训完，值得跑 45 轮。
# 4) in22k30 + swinb30 是本项目**第一个 bootstrap 区间不跨 0** 的组合
#    （valid 0.8112，Δ +0.0084，CI [+0.0011, +0.0158]）。
#
# 车道安排（三卡并行，按价值排序）：
#   A: in22k12（12 轮 448 强增强，22k 起点"落定版"）→ swin_b45（把没训完的训完）
#   B: in22k_384 → effv2m_384 → effv2s_384（三件套 384 交付版，T4 从 ~100ms 降到 ~90ms）
#   C: effv2m12（12 轮 448 强增强，与 in22k12 配对的另一半）→ cnx_small_384
#
# 用法：
#   cd ~/cxr/baseline3
#   tmux new -s n8
#   bash run_next8h.sh 2>&1 | tee runs/n8.log
#
# 可调：BUDGET_MIN=460  BS=12  WORKERS=6
# =============================================================================
set -u
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-12}"
BS="${BS:-12}"
WORKERS="${WORKERS:-6}"
BUDGET_MIN="${BUDGET_MIN:-460}"
STAMP=$(date +%m%d_%H%M)
LOG="runs/n8_$STAMP"
mkdir -p "$LOG"
START=$SECONDS

say() { echo "[$(date +%H:%M:%S)] $*"; }
remain() { echo $(( BUDGET_MIN - (SECONDS - START) / 60 )); }

SW22K="weights/convnext_base_22k_1k_224.pth"
HAS22K=0
[ -f "$SW22K" ] && [ "$(stat -c%s "$SW22K" 2>/dev/null || echo 0)" -gt 300000000 ] && HAS22K=1

GPUS=()
while IFS= read -r g; do [ -n "$g" ] && GPUS+=("$g"); done < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, '$2 < 1500 {print $1}' | head -3
)
if [ "${#GPUS[@]}" -eq 0 ]; then say "✗ 没有空闲 GPU（<1.5GB）"; exit 1; fi
say "可用 GPU：${GPUS[*]}｜预算 ${BUDGET_MIN} 分钟｜batch ${BS}"
say "IN-22k 权重：$([ "$HAS22K" -eq 1 ] && echo ✓ || echo ✗ 缺失，相关任务会跳过)"

lane() {
  local gpu="$1"; shift
  for spec in "$@"; do
    local name="${spec%%|*}"; local rest="${spec#*|}"
    local est="${rest%%|*}"; local cmd="${rest#*|}"
    if [ -f "runs/$name/best.pth" ]; then say "  [G$gpu] 跳过 ${name}（已存在）"; continue; fi
    if [ "$(remain)" -lt "$est" ]; then say "  [G$gpu] 跳过 ${name}（剩余 $(remain) < ${est}）"; continue; fi
    say "  [G$gpu] ▶ ${name}（预计 ${est} 分钟，剩余 $(remain)）"
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash -c "$cmd" > "$LOG/$name.log" 2>&1
    local rc=$? used=$(( (SECONDS - t0) / 60 ))
    [ "$rc" -eq 0 ] && say "  [G$gpu] ✓ ${name}（${used} 分钟）" \
                    || say "  [G$gpu] ✗ ${name}（码 ${rc}，${used} 分钟）见 $LOG/$name.log"
  done
}

A=(); B=(); C=()
if [ "$HAS22K" -eq 1 ]; then
  A+=("in22k12|115|python -u train3.py --arch convnext_base_in22k --pretrained $SW22K --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs 12 --seed 48 --aug strong --mixup 0.2 --out runs/in22k12")
  B+=("in22k_384|110|python -u train3.py --arch convnext_base_in22k --pretrained $SW22K --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs 12 --seed 48 --aug mild --out runs/in22k_384")
fi
A+=("swin_b45|300|python -u train3.py --arch swin_b --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs 45 --seed 48 --aug mild --out runs/swin_b45")
B+=("effv2m_384|110|python -u train3.py --arch efficientnet_v2_m --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs 12 --seed 48 --aug mild --out runs/effv2m_384")
B+=("effv2s_384|90|python -u train3.py --arch efficientnet_v2_s --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs 12 --seed 48 --aug mild --out runs/effv2s_384")
C+=("effv2m12|115|python -u train3.py --arch efficientnet_v2_m --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs 12 --seed 48 --aug strong --mixup 0.2 --out runs/effv2m12")
C+=("cnx_small_384|120|python -u train3.py --arch convnext_small --img-size 384 --batch-size $BS --lr auto --workers $WORKERS --epochs 12 --seed 48 --aug mild --out runs/cnx_small_384")

say "【阶段 1】三车道启动"
PIDS=()
N=$(( ${#GPUS[@]} < 3 ? ${#GPUS[@]} : 3 ))
[ "$N" -ge 1 ] && [ "${#A[@]}" -gt 0 ] && { lane "${GPUS[0]}" "${A[@]}" & PIDS+=("$!"); }
[ "$N" -ge 2 ] && [ "${#B[@]}" -gt 0 ] && { lane "${GPUS[1]}" "${B[@]}" & PIDS+=("$!"); }
[ "$N" -ge 3 ] && [ "${#C[@]}" -gt 0 ] && { lane "${GPUS[2]}" "${C[@]}" & PIDS+=("$!"); }
for p in "${PIDS[@]}"; do wait "$p"; done
say "【阶段 1】结束（$(( (SECONDS - START) / 60 )) 分钟）"

say "【阶段 2】valid / test 预测"
G0="${GPUS[0]}"
for name in in22k12 swin_b45 in22k_384 effv2m_384 effv2s_384 effv2m12 cnx_small_384; do
  [ -f "runs/$name/best.pth" ] || continue
  [ -f "runs/$name/pv.npz" ] || CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split valid \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pv.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pt.npz" ] || CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split test \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pt.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pt.npz" ] && say "  ✓ $name" || say "  ✗ $name"
done

say "【阶段 3】峰值汇总"
python - "$LOG" <<'PY' | tee "$LOG/summary.txt"
import json, glob
rows = []
for d in sorted(glob.glob("runs/*/metrics.json")):
    n = d.split("/")[1]
    if not any(k in n for k in ("in22k12", "swin_b45", "_384", "effv2m12")):
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
print("\n384 组的意义：推理耗时约为 448 的 0.735 倍，T4 折算 ~100ms → ~90ms。")
print("若 T4 实测 448 超标，改用 384 版交付（模型与提交都要换、要重新提交）。")
PY

echo
say "结束（$(( (SECONDS - START) / 60 )) 分钟）｜看 $LOG/summary.txt"
echo "=============================================================="
