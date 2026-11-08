#!/usr/bin/env bash
# ============================================================
# migrate.sh —— 把本项目迁到另一台设备
#
# 用法：
#   bash migrate.sh user@新设备:~/计挑赛          # 推到远程（走 ssh）
#   bash migrate.sh /Volumes/移动硬盘/计挑赛      # 推到本地盘/U 盘
#   DRY=1 bash migrate.sh user@host:~/计挑赛      # 只演练，不真传
#   SKIP_RAW=1 bash migrate.sh user@host:~/计挑赛 # 跳过 train+valid 原图（省 48G），test 仍传
#
# 特性：
#   - 分阶段传输，先传代码（几 MB），立刻就能在新设备上做语法/架构校验
#   - rsync 幂等：中断了再跑一次即可续传（--partial）
#   - 排除清单写死在脚本里，不靠记忆
# ============================================================
set -uo pipefail

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

SRC_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC_LOCAL"

# 目标端补一个斜杠（rsync 语义：有斜杠=拷内容）
DEST="${DEST%/}/"

# ---------- 传输开关 ----------
RSYNC_OPTS=(-a --partial --human-readable --progress)
[[ "${DRY:-0}" == "1" ]] && RSYNC_OPTS+=(--dry-run) && echo ">>> DRY-RUN 模式：只显示会传什么，不真传"
EXTRA=""     # 单次调用的额外开关（空格分隔；选项本身不含空格）

# ---------- 排除清单：这些在新设备上不需要 ----------
EXCLUDES=(
  --exclude='.git/'                       # 新设备用 git clone 拿代码，或删掉本目录后 clone
  --exclude='__pycache__/'                # 42 个 .pyc，会自动重建
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='._*'
  --exclude='results/cxr_t4.tgz'          # 647M，T4 核验包的压缩副本，pack_t4.sh 可重建
  --exclude='results/cxr_t4.zip'          # 647M，同上（第二份）
  --exclude='results/cxr_t4/weights/'     # 552M，与 results/weights_7767 逐字节相同
  --exclude='results/cxr_t4/baseline1/'   # 4.2M，baseline1/ 的完整副本
  --exclude='*.log'
  --exclude='nohup.out'
)

run_rsync() {
  local label="$1"; shift
  echo
  echo "============================================================"
  echo ">>> $label"
  echo "============================================================"
  # $EXTRA 故意不加引号：要按空格拆成多个选项；为空时展开成空，bash 3.2 + set -u 下安全
  rsync "${RSYNC_OPTS[@]}" $EXTRA "${EXCLUDES[@]}" "$@" || {
    echo "!! 上一步 rsync 返回非 0（常见原因：ssh 抖动/磁盘满），可修好后重跑本脚本续传"
    EXTRA=""
    return 1
  }
  EXTRA=""
}

echo "源目录：$SRC_LOCAL"
echo "目标  ：$DEST"
du -sh . 2>/dev/null | awk '{print "本机总占用："$1}'

# ---------- 阶段 1：代码 + 文档 + 小体积证据（几 MB）----------
# 注意：只排除 cxr_first 下的三个 split 子目录，**保留 3 个标注 CSV**
# （训练读标签全靠它们；排掉整个 cxr_first/ 会让 SKIP_RAW 模式直接起不来）
mkdir -p "$SRC_LOCAL/results" 2>/dev/null
run_rsync "阶段 1/5  代码、规则文档、标注 CSV、实验证据（小、快，先打通）" \
  --exclude='cxr_first/train/' --exclude='cxr_first/valid/' --exclude='cxr_first/test/' \
  --exclude='baseline/data600/' --exclude='dl/' \
  --exclude='results/weights_7767/' --exclude='results/cxr_t4/' \
  ./ "$DEST"

# ---------- 阶段 2：预处理数据 600²（2.5G）----------
run_rsync "阶段 2/5  预处理图 baseline/data600（训练直接吃它，33984 张）" \
  baseline/data600/ "${DEST}baseline/data600/"

# ---------- 阶段 3：权重（875M）----------
run_rsync "阶段 3/5  权重：77.67 双模型 + ImageNet-22k 起点" \
  results/weights_7767/ "${DEST}results/weights_7767/"
run_rsync "阶段 3/5  权重：dl/convnext_base_22k_1k_224.pth" \
  dl/ "${DEST}dl/"

# ---------- 阶段 4：T4 核验用的 105 张真原图（149M）----------
# 这两个目录现在只在 results/cxr_t4/ 里有，baseline1/ 下已不存在。
# 放回 baseline1/ 是为了让 cost.py / pack_t4.sh 的默认路径继续成立。
run_rsync "阶段 4/5  cost.py 计时用图（100 张 + 5 张预热）" \
  results/cxr_t4/pictures_100/ "${DEST}baseline1/pictures_100/"
run_rsync "阶段 4/5  5 张预热图" \
  results/cxr_t4/pictures_5/ "${DEST}baseline1/pictures_5/"

# ---------- 阶段 5：原始数据（52G，最慢，放最后）----------
# SKIP_RAW=1 只跳过 train(46G)+valid(2.1G)，**仍传 test(3.4G)**：
# cost.py 交付核验必须用真原图计时，test 原图不能省。
if [[ "${SKIP_RAW:-0}" == "1" ]]; then
  echo
  echo ">>> 阶段 5/5  跳过 train(46G)/valid(2.1G)，只传 test 原图"
  echo "    影响：不能重做 1024²/384 等换源分辨率预处理；训练/推理/核验完全不受影响。"
  EXTRA="--whole-file"      # 33984 张小 jpg，跳过增量滚动校验，快很多
  run_rsync "阶段 5/5  原始数据 cxr_first/test（cost.py 核验必需）" \
    cxr_first/test/ "${DEST}cxr_first/test/"
else
  echo
  echo ">>> 阶段 5/5  原图共 33984 张，单张只有 1~2MB："
  echo "    用 --whole-file 整文件传（rsync 默认为每个文件算增量，对这种小图纯浪费）。"
  EXTRA="--whole-file"
  run_rsync "阶段 5/5  原始数据 cxr_first（train 46G + valid 2.1G + test 3.4G + 3 个 CSV）" \
    cxr_first/ "${DEST}cxr_first/"
fi

# ---------- 收尾提示 ----------
cat <<EOF

============================================================
迁移完成。若目标是「远程主机」，还需在新设备上做 3 件事：
============================================================

# ① 恢复 pack_t4.sh 找权重的路径（用软链，不额外占 537M）
cd ~/计挑赛/baseline1
mkdir -p runs/full_base runs/full_effv2m
ln -sf ../../../results/weights_7767/full_base.pth   runs/full_base/best.pth
ln -sf ../../../results/weights_7767/full_effv2m.pth runs/full_effv2m/best.pth
cp ../results/submission_7767_ens.csv submission_ens.csv

# ② 重建丢失的五折划分（原文件已不存在，种子固定 48 可复现同一份）
cd ~/计挑赛/baseline1 && python make_folds.py --k 5 --seed 48 --out runs/folds

# ③ 校验（见 README_迁移与复现.md 第 4 节，7 条全绿再开训）

============================================================
本次未迁移的内容（合计约 1.85G，都可重建）：
  results/cxr_t4.tgz          647M  → bash pack_t4.sh 重建
  results/cxr_t4.zip          647M  → 同上，两份压缩副本纯重复
  results/cxr_t4/weights/     552M  → 与 results/weights_7767 完全相同
  results/cxr_t4/baseline1/   4.2M  → baseline1/ 的副本
  __pycache__ 42 个 .pyc      ~400K → 自动重建
  .git/                       —— → 新设备 git clone 即可（已瘦到 788K）
============================================================
EOF

if [[ "${SKIP_RAW:-0}" == "1" ]]; then
cat <<'EOF'

============================================================
本次按 SKIP_RAW=1 额外跳过了：cxr_first/train（46G）+ cxr_first/valid（2.1G）
  想补传（以后要重做 1024²/384 预处理时）：
    rsync -av --partial --progress cxr_first/train/  目标:~/计挑赛/cxr_first/train/
    rsync -av --partial --progress cxr_first/valid/  目标:~/计挑赛/cxr_first/valid/
============================================================
EOF
fi
