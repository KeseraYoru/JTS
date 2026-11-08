#!/usr/bin/env bash
# =============================================================================
# 5 小时无人值守训练（baseline3）
#
# 目的：把「老师建议里唯一合规且没试过的一枪」跑掉 —— 强增强 + Mixup，
#       并在同一个 fold0 上做单变量 A/B 对照，胜出者自动跑全量。
#
# 设计原则：
#   1) 一切单变量：三组 fold 跑用同一份代码、同一折、同一 batch、同一 LR 规则；
#   2) 判据前置：Δ ≥ +0.003 才换（fold0 有 2211 患者，噪声 ±0.0044）；
#   3) 时间闸门：剩余时间不够就跳过，不会留下半截任务；
#   4) 可断可续：每步先看 runs/<name>/best.pth，存在即跳过；重跑同一命令即可；
#   5) 不抢别人的卡：自动挑空闲显存最多的 GPU，且要求 ≥ 9GB，只在单卡上跑。
#
# 用法（服务器上）：
#   cd ~/cxr/baseline3
#   tmux new -s auto
#   bash run_5h.sh 2>&1 | tee runs/auto_$(date +%m%d_%H%M).log
#   # Ctrl-B 再 D 脱离
#
# 可调环境变量：
#   BUDGET_MIN=285   总预算（分钟），默认 285（5 小时留 15 分钟收尾）
#   BS=12            单卡 batch（显存约 8GB；若卡上空闲更多可设 16）
#   WORKERS=8        DataLoader 进程数（mtt 在用 CPU，别开太大）
# =============================================================================
set -u
cd "$(dirname "$0")"

BUDGET_MIN="${BUDGET_MIN:-285}"
BS="${BS:-12}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-12}"
FOLDS="$(cd ../baseline1/runs/folds && pwd)"
SSL="$(cd ../baseline1/runs/ssl && pwd)/simclr200_384.pth"
STAMP=$(date +%Y%m%d_%H%M)
LOG="runs/auto_$STAMP"
mkdir -p "$LOG"
START=$SECONDS

say() { echo "[$(date +%H:%M:%S)] $*"; }
remain() { echo $(( BUDGET_MIN - (SECONDS - START) / 60 )); }

# ---------- 选卡：空闲显存最多的那张，要求 ≥ 9GB ----------
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, '$2 >= 9000 {print $2" "$1}' \
    | sort -rn | head -1 | cut -d' ' -f2
}

wait_gpu() {          # 最多等 20 分钟
  local waited=0 g
  while [ $waited -lt 1200 ]; do
    g=$(pick_gpu)
    if [ -n "$g" ]; then echo "$g"; return 0; fi
    say "  当前没有 ≥9GB 空闲的卡，等 2 分钟再看（已等 $((waited/60)) 分钟）"
    sleep 120; waited=$((waited + 120))
  done
  echo ""
}

# ---------- 跑一个任务 ----------
# run_job <名字> <预计分钟> <命令...>
run_job() {
  local name="$1" est="$2"; shift 2
  local out="runs/$name"

  if [ -f "$out/best.pth" ]; then
    say "跳过 $name：已存在 $out/best.pth（断点续跑）"
    return 0
  fi
  if [ "$(remain)" -lt "$est" ]; then
    say "跳过 $name：剩余 $(remain) 分钟 < 预计 ${est} 分钟"
    return 1
  fi

  local gpu; gpu=$(wait_gpu)
  if [ -z "$gpu" ]; then
    say "跳过 $name：等了 20 分钟仍无可用 GPU"
    return 1
  fi

  say "▶ 开始 $name ｜GPU$gpu｜batch $BS｜预计 ${est} 分钟｜剩余 $(remain) 分钟"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$@" 2>&1 | tee "$LOG/$name.log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -ne 0 ]; then
    if grep -qi "out of memory" "$LOG/$name.log"; then
      say "✗ $name 显存不足（OOM）。建议重跑时把 BS 降到 8：BS=8 bash run_5h.sh"
    else
      say "✗ $name 失败（退出码 $rc），详见 $LOG/$name.log 末尾"
    fi
    return 1
  fi
  say "✓ 完成 $name（累计已用 $(( (SECONDS - START) / 60 )) 分钟，剩余 $(remain) 分钟）"
  return 0
}

# ---------- 从 metrics.json 读峰值 ----------
peak() {
  python - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    b = max(d, key=lambda x: x["study_auc"])
    print(f'{b["study_auc"]:.4f} {b["epoch"]}')
except Exception:
    print("0.0000 -")
PY
}

echo "=============================================================="
say "预算 ${BUDGET_MIN} 分钟 | batch ${BS} | epochs ${EPOCHS} | 日志 $LOG"
say "折文件 $FOLDS"
say "SSL 权重 $SSL"
[ -f "$SSL" ] && say "  SSL 存在 ✓" || say "  SSL 缺失 ✗（会退回纯 ImageNet 起点，注意台账要注明）"
echo "=============================================================="

# =============================================================================
# 阶段 1：fold0 上的 A/B 对照（同患者、同代码、同超参，只差增强）
# =============================================================================
say "【阶段 1】fold0 增强对照"

run_job f0_ssl_mild 55 \
  python -u train3.py --arch convnext_base --img-size 448 --batch-size "$BS" \
    --lr auto --workers "$WORKERS" --epochs "$EPOCHS" --seed 48 \
    --aug mild --ssl "$SSL" \
    --train-csv "$FOLDS/fold_0_train.csv" --val-csv "$FOLDS/fold_0_val.csv" \
    --out runs/f0_ssl_mild

run_job f0_ssl_strong 55 \
  python -u train3.py --arch convnext_base --img-size 448 --batch-size "$BS" \
    --lr auto --workers "$WORKERS" --epochs "$EPOCHS" --seed 48 \
    --aug strong --mixup 0.2 --ssl "$SSL" \
    --train-csv "$FOLDS/fold_0_train.csv" --val-csv "$FOLDS/fold_0_val.csv" \
    --out runs/f0_ssl_strong

# ---------- 判定：强增强是否值得进全量 ----------
AUG_WIN="mild"
if [ -f runs/f0_ssl_mild/metrics.json ] && [ -f runs/f0_ssl_strong/metrics.json ]; then
  read -r M_AUC M_EP <<<"$(peak runs/f0_ssl_mild/metrics.json)"
  read -r S_AUC S_EP <<<"$(peak runs/f0_ssl_strong/metrics.json)"
  AUG_WIN=$(python -c "print('strong' if $S_AUC - $M_AUC >= 0.003 else 'mild')")
  say "mild 峰值 $M_AUC (ep$M_EP) ｜ strong 峰值 $S_AUC (ep$S_EP) ｜ Δ $(python -c "print(f'{$S_AUC-$M_AUC:+.4f}')")"
  say "→ 增强档位判定：$AUG_WIN（门槛 Δ ≥ +0.003）"
fi
echo "$AUG_WIN" > "$LOG/aug_winner.txt"

# =============================================================================
# 阶段 2：用胜出的增强档跑全量（30344 张），产出可用于提交的模型
# =============================================================================
say "【阶段 2】全量训练（aug=$AUG_WIN）"
if [ "$AUG_WIN" = "strong" ]; then
  run_job full_aug_winner 75 \
    python -u train3.py --arch convnext_base --img-size 448 --batch-size "$BS" \
      --lr auto --workers "$WORKERS" --epochs "$EPOCHS" --seed 48 \
      --aug strong --mixup 0.2 --ssl "$SSL" --out runs/full_aug_winner
else
  run_job full_aug_winner 75 \
    python -u train3.py --arch convnext_base --img-size 448 --batch-size "$BS" \
      --lr auto --workers "$WORKERS" --epochs "$EPOCHS" --seed 48 \
      --aug mild --ssl "$SSL" --out runs/full_aug_winner
fi

# =============================================================================
# 阶段 3：ViewPosition 辅助头（零推理成本，规则允许，从没用过）
# =============================================================================
say "【阶段 3】ViewPosition 辅助头"
run_job f0_ssl_view 60 \
  python -u train3.py --arch convnext_base --img-size 448 --batch-size "$BS" \
    --lr auto --workers "$WORKERS" --epochs "$EPOCHS" --seed 48 \
    --aug "$AUG_WIN" --use-view --ssl "$SSL" \
    --train-csv "$FOLDS/fold_0_train.csv" --val-csv "$FOLDS/fold_0_val.csv" \
    --out runs/f0_ssl_view

# 若 ViewPosition 也有收益且还有时间，跑一版全量
if [ -f runs/f0_ssl_mild/metrics.json ] && [ -f runs/f0_ssl_view/metrics.json ]; then
  read -r V_AUC _ <<<"$(peak runs/f0_ssl_view/metrics.json)"
  read -r M_AUC _ <<<"$(peak runs/f0_ssl_mild/metrics.json)"
  VIEW_GAIN=$(python -c "print(f'{$V_AUC-$M_AUC:+.4f}')")
  say "ViewPosition Δ vs mild = $VIEW_GAIN"
  if python -c "import sys; sys.exit(0 if $V_AUC - $M_AUC >= 0.003 else 1)"; then
    run_job full_view 80 \
      python -u train3.py --arch convnext_base --img-size 448 --batch-size "$BS" \
        --lr auto --workers "$WORKERS" --epochs "$EPOCHS" --seed 48 \
        --aug "$AUG_WIN" --use-view --ssl "$SSL" --out runs/full_view
  fi
fi

# =============================================================================
# 阶段 4：收尾 —— 出预测 + 配对 bootstrap + 汇总
# =============================================================================
say "【阶段 4】生成折内预测并做配对检验"

for n in f0_ssl_mild f0_ssl_strong f0_ssl_view; do
  [ -f "runs/$n/best.pth" ] || continue
  [ -f "runs/$n/pv.npz" ] && continue
  python -u predict3.py --split train --csv "$FOLDS/fold_0_val.csv" \
    --ckpt "runs/$n/best.pth" --out "runs/$n/pv.npz" >>"$LOG/predict.log" 2>&1 \
    && say "  ✓ $n 折内预测已出" || say "  ✗ $n 预测失败（见 $LOG/predict.log）"
done

if [ -f runs/f0_ssl_mild/pv.npz ] && [ -f runs/f0_ssl_strong/pv.npz ]; then
  python ../baseline1/tools/bootstrap_ci.py \
    --a runs/f0_ssl_mild/pv.npz --b runs/f0_ssl_strong/pv.npz \
    --csv "$FOLDS/fold_0_val.csv" 2>&1 | tee "$LOG/bootstrap_aug.log" | tail -20
fi

# 全量模型在官方 valid 上出预测（供回来看，也供与 0.7997 对比）
if [ -f runs/full_aug_winner/best.pth ]; then
  python -u predict3.py --split valid --ckpt runs/full_aug_winner/best.pth \
    --out runs/full_aug_winner/pv_valid.npz >>"$LOG/predict.log" 2>&1 \
    && say "  ✓ full_aug_winner 官方 valid 预测已出"
fi

# ---------- 汇总 ----------
python - "$LOG" <<'PY' | tee "$LOG/summary.txt"
import json, os, sys, glob
log = sys.argv[1]
rows = []
for d in sorted(glob.glob("runs/*/metrics.json")):
    name = d.split("/")[1]
    try:
        m = json.load(open(d))
        b = max(m, key=lambda x: x["study_auc"])
        rows.append((name, b["study_auc"], b["epoch"], b.get("per_class_auc")))
    except Exception:
        pass
print(f"{'实验':<20}{'峰值':>8}{'峰值轮':>8}   类1     类4     类2")
print("-" * 62)
for name, auc, ep, cls in rows:
    if name.startswith(("f0_ssl", "full_aug", "full_view")):
        c = cls or []
        def g(i):
            return f"{c[i-1]:.3f}" if len(c) >= i else "  -  "
        print(f"{name:<20}{auc:>8.4f}{ep:>8}{g(1):>8}{g(4):>8}{g(2):>8}")
print("\n参照：baseline1 full_base 0.7997 ｜ 双模型集成 0.8027（平台 77.67）")
print(f"\n日志目录：{log}")
PY

echo
say "全部结束。剩余预算 $(remain) 分钟。"
say "重点看这三样："
say "  1) $LOG/summary.txt          —— 各实验峰值与关键类"
say "  2) $LOG/aug_winner.txt       —— 强增强是否胜出"
say "  3) $LOG/bootstrap_aug.log    —— 配对差值的 95% 区间"
echo "=============================================================="
