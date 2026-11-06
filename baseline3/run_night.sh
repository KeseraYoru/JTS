#!/usr/bin/env bash
# 今夜挂机队列：ImageNet-22k ConvNeXt（MIT 许可，合规）
#
# 为什么是它：同架构、同配方、同尺寸，只把预训练数据从 ImageNet-1k 换成 22k ——
# 单变量最干净，且不需要写任何新代码（vendor 架构已就绪、自检通过）。
#
# 设计原则：**每一步都可跳过**（已有产物就跳过），这样断线重跑不会白烧时间。
set -u
cd "$(dirname "$0")"

F=../baseline1/runs/folds
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p runs

# 找权重：优先 22k→1k 微调版（迁移更好、体积更小），其次纯 22k 版。
# 并且**必须本地传路径**给 train3：让它别自己联网下（服务器 CA 过期，urllib 会直接失败）。
W=""
for f in weights/convnext_base_22k_1k_224.pth weights/convnext_base_22k_224.pth; do
  if [ -s "$f" ]; then
    mb=$(( $(wc -c < "$f") / 1024 / 1024 ))
    if [ "$mb" -ge 300 ]; then W="$f"; fi
  fi
done

if [ -z "$W" ]; then
  echo "✗ 没找到完整权重（需要 ≥300MB）。下载方法："
  echo "  cd ~/cxr/baseline3/weights"
  echo "  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \\"
  echo "    curl -k -L --http1.1 --retry 5 -C - \\"
  echo "    -o convnext_base_22k_1k_224.pth \\"
  echo "    https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth"
  echo "  ls -lh convnext_base_22k_1k_224.pth    # 必须约 338MB；不够就再跑一次（会续传）"
  exit 1
fi
echo "使用权重: $W（$(( $(wc -c < "$W") / 1024 / 1024 ))MB）"

# 统一的训练入口；产物已存在则跳过（断线可续）
run () {
  local name="$1"; shift
  if [ -f "runs/$name/best.pth" ]; then
    echo "== 跳过 $name（已有 best.pth）=="; return 0
  fi
  echo "== 开始 $name  $(date '+%H:%M:%S') =="
  python -u train3.py "$@" --out "runs/$name" 2>&1 | tee "runs/$name.log"
  echo "== 结束 $name  $(date '+%H:%M:%S') =="
}

# ① 探针：2 轮，回答"22k 起点是否更强"（约 6~10 分钟）
run probe_22k --arch convnext_base_in22k --img-size 448 --batch-size 16 \
  --lr 1e-4 --probe --pretrained "$W" \
  --train-csv "$F/fold_0_train.csv" --val-csv "$F/fold_0_val.csv"

# ② 三个 seed 全量训练（每个约 30 分钟，合计 ~90 分钟）
for s in 48 49 50; do
  run "in22k_s$s" --arch convnext_base_in22k --img-size 448 --batch-size 16 \
    --lr 1e-4 --epochs 12 --seed "$s" --pretrained "$W"
done

# ③ 三个 seed 做权重平均（soup），并各自在官方 valid 上出预测
if [ -f runs/in22k_s48/best.pth ] && [ -f runs/in22k_s49/best.pth ] && [ -f runs/in22k_s50/best.pth ]; then
  python tools/soup3.py --ckpt runs/in22k_s48/best.pth runs/in22k_s49/best.pth \
    runs/in22k_s50/best.pth --out runs/in22k_soup.pth
  for n in in22k_s48 in22k_s49 in22k_s50 in22k_soup; do
    python -u predict3.py --split valid --ckpt "runs/$n/best.pth" \
      --out "runs/pv_$n.npz" 2>/dev/null || true
  done
  # soup 的 ckpt 不在 <dir>/best.pth，单独出一份
  python -u predict3.py --split valid --ckpt runs/in22k_soup.pth --out runs/pv_in22k_soup.npz
fi

# ④ 汇总
echo
echo "=========== 今夜结果 ==========="
python - <<'EOF'
import json, glob, os
for p in sorted(glob.glob("runs/*/metrics.json")):
    name = p.split("/")[1]
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not d:
        continue
    b = max(d, key=lambda x: x["study_auc"])
    print(f"{name:<14} 峰值 {b['study_auc']:.4f} @ep{b['epoch']:<2} | "
          f"末轮 {d[-1]['study_auc']:.4f}")
print()
print("对照：convnext_base(ImageNet-1k) 全量 valid = 0.7997；effv2m = 0.7993；"
      "双集成 = 0.8027（平台 77.67）")
EOF
echo
echo "下一步（明天）："
echo "  python predict3.py --split test --ckpt runs/in22k_soup.pth --out runs/pt_22k.npz"
echo "  python ensemble3.py --npz runs/pt_22k.npz <effv2m 的 npz> --method mean --out submission_22k.csv"
