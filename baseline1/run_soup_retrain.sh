#!/usr/bin/env bash
# =============================================================================
# 重训 cnx + effv2m，并测试「最后几轮 EMA 平均（SWA）」是否优于 best.pth
#
# 背景（为什么做这个）：
#   五次提交证明 valid 排序对平台没有预测力 —— valid 最高的两个（0.8059/0.8053）
#   平台只有 77.37/77.18，而 valid 中游的 0.8027 是 77.67。
#   best.pth 是在 859 个 study 上挑出来的"运气峰"；ep12 的 cosine 已把 LR 退到 0，
#   权重才真正落定。相邻几轮的 EMA 平均（SWA）方差更小，对分布外的 test 通常更稳。
#
# 合规性：
#   - 不做任何"看平台分再挑 ckpt"的动作（那是规则第 5 条禁止的）；
#   - 采用哪一轮由**预先写死的规则**决定（见下方 DECIDE），不需要测试集信息；
#   - 规则第 3 条允许用验证集做 ckpt 选择（我们从未把 valid 并入训练）。
#
# 用法：
#   cd ~/cxr/baseline1
#   tmux new -s soup
#   bash run_soup_retrain.sh 2>&1 | tee runs/soup_retrain.log
#
# 可调：BUDGET_MIN=200  BS=12  WORKERS=8
# =============================================================================
set -u
cd "$(dirname "$0")"

BUDGET_MIN="${BUDGET_MIN:-200}"
BS="${BS:-12}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-12}"
STAMP=$(date +%m%d_%H%M)
LOG="runs/soup_$STAMP"
mkdir -p "$LOG"
START=$SECONDS

say() { echo "[$(date +%H:%M:%S)] $*"; }
remain() { echo $(( BUDGET_MIN - (SECONDS - START) / 60 )); }

pick_gpu() {
  local need="${1:-9000}"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, -v n="$need" '$2 >= n {print $2" "$1}' \
    | sort -rn | head -1 | cut -d' ' -f2
}

wait_gpu() {
  local need="$1" waited=0 g
  while [ $waited -lt 1200 ]; do
    g=$(pick_gpu "$need")
    [ -n "$g" ] && { echo "$g"; return 0; }
    say "  暂无 >=$((need/1024))GB 空闲的卡，等 2 分钟（已等 $((waited/60)) 分钟）"
    sleep 120; waited=$((waited + 120))
  done
  echo ""
}

# run_job <名字> <预计分钟> <命令...>
run_job() {
  local name="$1" est="$2"; shift 2
  local out="runs/$name"

  if [ -f "$out/best.pth" ] && ls "$out"/ema_f*.pth >/dev/null 2>&1; then
    say "跳过 $name：best.pth 与 ema_f*.pth 都在（断点续跑）"
    return 0
  fi
  if [ "$(remain)" -lt "$est" ]; then
    say "跳过 $name：剩余 $(remain) 分钟 < 预计 ${est} 分钟"
    return 1
  fi

  local gpu; gpu=$(wait_gpu 9000)
  if [ -z "$gpu" ]; then say "跳过 $name：无 >=9GB 空闲的卡"; return 1; fi

  say "▶ $name ｜GPU$gpu｜batch $BS｜预计 ${est} 分钟｜剩余 $(remain) 分钟"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$@" 2>&1 | tee "$LOG/$name.log"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -ne 0 ]; then
    if grep -qi "out of memory" "$LOG/$name.log"; then
      say "✗ $name OOM。重跑请用 BS=8"
    else
      say "✗ $name 失败（$rc），见 $LOG/$name.log"
    fi
    return 1
  fi
  say "✓ 完成 $name（累计 $(( (SECONDS - START) / 60 )) 分钟，剩余 $(remain) 分钟）"
  return 0
}

# 取某个 run 的最后 N 轮权重路径（按 epoch 数字排序）
last_epochs() {          # $1=run 目录 $2=取几轮
  ls "$1"/ema_f*.pth 2>/dev/null \
    | sed 's/.*ema_f\([0-9]*\)\.pth/\1 &/' \
    | sort -n | tail -"$2" | awk '{print $2}'
}

echo "=============================================================="
say "预算 ${BUDGET_MIN} 分钟 ｜ batch ${BS} ｜ epochs ${EPOCHS} ｜ 日志 $LOG"
echo "=============================================================="

# =============================================================================
# 阶段 1：重训两个成员（保留每轮 EMA —— 这次绝不能删）
# =============================================================================
say "【阶段 1】重训两个成员"

run_job soup_cnx 80 \
  python -u train.py --backbone convnext_base --epochs "$EPOCHS" --img-size 448 \
    --batch-size "$BS" --lr 1.1e-4 --workers "$WORKERS" --seed 48 \
    --ssl runs/ssl/simclr200_384.pth --out runs/soup_cnx

run_job soup_eff 80 \
  python -u train.py --backbone efficientnet_v2_m --epochs "$EPOCHS" --img-size 448 \
    --batch-size "$BS" --lr 1.1e-4 --workers "$WORKERS" --seed 48 \
    --out runs/soup_eff

# =============================================================================
# 阶段 2：对每个成员做 soup（最后 2 轮 / 最后 3 轮）
# =============================================================================
say "【阶段 2】构造 SWA 权重"

SOUP="python ../baseline3/tools/soup3.py"
for run in soup_cnx soup_eff; do
  for k in 2 3; do
    # 用数组而不是字符串，避免依赖 shell 的分词行为（zsh 不会按换行拆；mapfile 在 bash 3.2 没有）
    files=()
    while IFS= read -r f; do [ -n "$f" ] && files+=("$f"); done < <(last_epochs "runs/$run" "$k")
    cnt=${#files[@]}
    if [ "$cnt" -lt 2 ]; then
      say "  跳过 $run 的 last$k：可用轮数只有 $cnt"
      continue
    fi
    out="runs/$run/soup_last$k.pth"
    if [ -f "$out" ]; then say "  已存在 $out"; continue; fi
    $SOUP --ckpt "${files[@]}" --out "$out" >>"$LOG/soup.log" 2>&1 \
      && say "  ✓ $out（平均 $cnt 轮）" || say "  ✗ $out 失败"
  done
done

# =============================================================================
# 阶段 3：valid 上评估三种形态，并按预注册规则选一个
# =============================================================================
say "【阶段 3】valid 评估（best / last2 / last3）"

evaluate_pair() {        # $1=候选后缀（空/best、_last2、_last3） $2=输出 npz
  local suffix="$1" outnpz="$2"
  local a="runs/soup_cnx/best.pth" b="runs/soup_eff/best.pth"
  [ -n "$suffix" ] && { a="runs/soup_cnx/soup${suffix}.pth"; b="runs/soup_eff/soup${suffix}.pth"; }
  [ -f "$a" ] && [ -f "$b" ] || { echo "0.0000"; return; }
  python -u predict.py --split valid --ckpt "$a" "$b" --out "$outnpz" 2>/dev/null \
    | grep "study 级 Macro AUC" | tail -1 | awk '{print $NF}'
}

A_BEST=$(evaluate_pair ""       runs/pv_soup_best.npz)
A_L2=$(evaluate_pair "_last2"   runs/pv_soup_last2.npz)
A_L3=$(evaluate_pair "_last3"   runs/pv_soup_last3.npz)
say "  best  $A_BEST"
say "  last2 $A_L2"
say "  last3 $A_L3"

# ---- 预注册决策规则（写死，事后不改） ----
# 1) 默认用 last3（理论最稳：方差最小、LR 已退火）；
# 2) 若 last3 相对 best 掉超过 0.005 → 说明平均抹掉了信息，退回 best；
# 3) last2 只作为参考，不参与选择（少一个选择维度 = 少一分挑选偏差）。
DECIDE=$(python -c "
b=float('$A_BEST' or 0); l3=float('$A_L3' or 0)
print('last3' if (b>0 and l3>=b-0.005) else 'best')
" 2>/dev/null || echo best)
[ "$DECIDE" = "last3" ] && SUF="_last3" || SUF=""
say "  → 采用：$([ -z "$SUF" ] && echo 'best.pth' || echo 'soup_last3')"
echo "$DECIDE" > "$LOG/decision.txt"

# =============================================================================
# 阶段 4：出 test 提交 + 合规核验
# =============================================================================
say "【阶段 4】出提交并核验"

CK_A="runs/soup_cnx/best.pth"; CK_B="runs/soup_eff/best.pth"
[ -n "$SUF" ] && { CK_A="runs/soup_cnx/soup${SUF}.pth"; CK_B="runs/soup_eff/soup${SUF}.pth"; }

if [ -f "$CK_A" ] && [ -f "$CK_B" ]; then
  say "  使用：$CK_A  +  $CK_B"
  python -u predict.py --split test --ckpt "$CK_A" --out runs/pt_soup_a.npz 2>&1 | tail -2
  python -u predict.py --split test --ckpt "$CK_B" --out runs/pt_soup_b.npz 2>&1 | tail -2
  python ensemble.py --split test --inputs runs/pt_soup_a.npz runs/pt_soup_b.npz \
    --out submission_soup.csv 2>&1 | tail -3
  wc -l submission_soup.csv
  say "  合规核验（耗时必须 ≈42ms、不一致必须 0 张）："
  python -u cost.py submission_soup.csv pictures_100 pictures_5 \
    --ckpt "$CK_A" "$CK_B" 2>&1 | tail -6
else
  say "  ✗ 找不到权重，跳过出提交"
fi

echo
say "结束（剩余预算 $(remain) 分钟）。回来看："
say "  $LOG/decision.txt      —— 最终用了 best 还是 last3"
say "  $LOG/soup.log          —— soup 构造记录"
say "  submission_soup.csv    —— 待提交文件"
echo "=============================================================="
