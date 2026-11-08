#!/usr/bin/env bash
# =============================================================================
# 三卡并行过夜训练（baseline3；GPU 空闲时用）
#
# 为什么用 baseline3 而不是 baseline1
# ----------------------------------
# baseline1 的 train.py **没有 --aug / --mixup**（只有 mild 增强写死在 dataset 里）。
# baseline3 才有 --aug {mild,strong} / --mixup，且 predict3 / ensemble3 / cost3 都验证过。
# 所以这批任务全部走 baseline3，产出物用它自己的工具链拼装。
#
# 为什么做这三个
# -------------
# 五个提交点已证明 **valid 排序对平台没有预测力**（0.8059→77.37，0.8027→77.67）。
# 所以目标不是"把 valid 刷高"，而是**产出几个彼此不同、且都能交付的候选**，用每天 2 次额度去撞。
#
#   GPU-A  cnx30_strong : convnext_base 30 轮 + 强增强 + Mixup
#           → 测"长训练 + 重增强"这个标准配方。我们只在 12 轮上试过强增强，那是欠训练。
#   GPU-B  eff30_strong : efficientnet_v2_m 30 轮 + 同配方
#           → 与 A 同配方不同架构，天然可配成一对，交付仍是 2 模型、耗时不变。
#   GPU-C  cnx_small30  : convnext_small 30 轮 + 常规增强
#           → 便宜的第三成员（94MB），用来增加候选对的数量。
#
# 交付约束：任何候选 ≤2 个成员（双模型已 39ms，T4 约 100ms，没有余量再加）。
#
# 用法：
#   cd ~/cxr/baseline3
#   tmux new -s n3
#   bash run_3gpu_night.sh 2>&1 | tee runs/n3.log
#
# 可调：EPOCHS=30 BS=12 WORKERS=6
# =============================================================================
set -u
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-30}"
BS="${BS:-12}"
WORKERS="${WORKERS:-6}"
STAMP=$(date +%m%d_%H%M)
LOG="runs/n3_$STAMP"
mkdir -p "$LOG"
START=$SECONDS

say() { echo "[$(date +%H:%M:%S)] $*"; }
SSL="../baseline1/runs/ssl/simclr200_384.pth"

# ---------- 空闲的卡（占用 <1GB），最多 3 张 ----------
GPUS=()
while IFS= read -r g; do [ -n "$g" ] && GPUS+=("$g"); done < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, '$2 < 1000 {print $1}' | head -3
)

if [ "${#GPUS[@]}" -eq 0 ]; then
  say "✗ 没有空闲 GPU（占用 <1GB）。等别人跑完再来。"
  exit 1
fi
say "可用 GPU：${GPUS[*]}（${#GPUS[@]} 张）"
[ -f "$SSL" ] && say "SSL 权重 ✓ $SSL" || say "⚠ SSL 权重缺失，convnext 会退回纯 ImageNet 起点（台账要注明）"

# ---------- 组任务（按可用卡数降级） ----------
TASKS=()
[ "${#GPUS[@]}" -ge 1 ] && TASKS+=("${GPUS[0]}|cnx30_strong|python -u train3.py --arch convnext_base --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug strong --mixup 0.2 --ssl $SSL --out runs/cnx30_strong")
[ "${#GPUS[@]}" -ge 2 ] && TASKS+=("${GPUS[1]}|eff30_strong|python -u train3.py --arch efficientnet_v2_m --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug strong --mixup 0.2 --out runs/eff30_strong")
[ "${#GPUS[@]}" -ge 3 ] && TASKS+=("${GPUS[2]}|cnx_small30|python -u train3.py --arch convnext_small --img-size 448 --batch-size $BS --lr auto --workers $WORKERS --epochs $EPOCHS --seed 48 --aug mild --out runs/cnx_small30")

say "【阶段 1】并行训练 ${#TASKS[@]} 个任务，各 $EPOCHS 轮（预计 3~3.5 小时）"
PIDS=()
for t in "${TASKS[@]}"; do
  gpu="${t%%|*}"; rest="${t#*|}"; name="${rest%%|*}"; cmd="${rest#*|}"

  if [ -f "runs/$name/best.pth" ]; then
    say "  跳过 ${name}（已有 best.pth，断点续跑）"
    continue
  fi

  say "  ▶ $name → GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -c "$cmd" > "$LOG/$name.log" 2>&1 &
  PIDS+=("$!:$name")
done

if [ "${#PIDS[@]}" -eq 0 ]; then
  say "  没有需要跑的任务（全部已存在）"
else
  for p in "${PIDS[@]}"; do
    pid="${p%%:*}"; name="${p#*:}"
    if wait "$pid"; then
      say "  ✓ ${name}（累计 $(( (SECONDS - START) / 60 )) 分钟）"
    else
      say "  ✗ $name 失败，见 $LOG/$name.log"
      grep -qi "out of memory" "$LOG/$name.log" 2>/dev/null && say "     → OOM，重跑请 BS=8"
    fi
  done
fi

# ---------- 阶段 2：valid + test 预测 ----------
say "【阶段 2】出预测"
G0="${GPUS[0]}"
for name in cnx30_strong eff30_strong cnx_small30; do
  [ -f "runs/$name/best.pth" ] || continue
  [ -f "runs/$name/pv.npz" ] || \
    CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split valid \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pv.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pt.npz" ] || \
    CUDA_VISIBLE_DEVICES="$G0" python -u predict3.py --split test \
      --ckpt "runs/$name/best.pth" --out "runs/$name/pt.npz" >>"$LOG/predict.log" 2>&1
  [ -f "runs/$name/pv.npz" ] && say "  ✓ $name 预测就绪" || say "  ✗ $name 预测失败（见 $LOG/predict.log）"
done

# ---------- 阶段 3：所有两两组合 ----------
say "【阶段 3】两两组合（valid 仅供记录）"
if [ -f ../baseline1/tools/pair_search.py ]; then
  python ../baseline1/tools/pair_search.py \
    --member full_base=../baseline1/runs/full_base/pv2.npz \
    --member full_effv2m=../baseline1/runs/full_effv2m/pv.npz \
    --member cnx30=runs/cnx30_strong/pv.npz \
    --member eff30=runs/eff30_strong/pv.npz \
    --member cnxsmall30=runs/cnx_small30/pv.npz \
    --champion full_base full_effv2m --n 1000 2>&1 | tee "$LOG/pairs.log" | tail -40
else
  say "  缺 ../baseline1/tools/pair_search.py，跳过"
fi

# ---------- 阶段 4：给最优的两个组合各出一份提交（不自动上传，只备好） ----------
say "【阶段 4】预备两份提交（各两成员，体积均 <450MB）"
if [ -f runs/cnx30_strong/pt.npz ] && [ -f runs/eff30_strong/pt.npz ]; then
  python -u ensemble3.py --npz runs/cnx30_strong/pt.npz runs/eff30_strong/pt.npz \
    --split test --method mean --out submission_30ep.csv >>"$LOG/ens.log" 2>&1 \
    && say "  ✓ submission_30ep.csv（cnx30 + eff30）" || say "  ✗ 组失败（见 $LOG/ens.log）"
fi

echo
say "结束（累计 $(( (SECONDS - START) / 60 )) 分钟）"
say "明天看：$LOG/pairs.log ｜ runs/*/best.pth ｜ submission_30ep.csv"
say "提醒：valid 不能选型。用 2 次/天额度去试，每次只交「2 成员 + 体积<450MB + cost 0 张不一致」的候选。"
echo "=============================================================="
