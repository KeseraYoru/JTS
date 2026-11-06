#!/usr/bin/env bash
# 打包 T4 核验包（在训练机上跑）
#
# 目的：把"当前最高分 77.67 那套东西"封成一个约 700MB 的 tar，传到免费 T4（Kaggle）
# 跑一次 cost.py，确认双模型在官方硬件上能否 ≤100ms。
#
# 为什么必须做：附录 A 要求包内 submission.csv = 历史最高分那份，且 cost.py 必须能复现它。
# 若双模型在 T4 上超时，77.67 就是"不可交付"的分数 —— 这比再涨 0.2 分重要得多。
#
# ⚠ 本包基于 baseline1（77.67 是它产出的）；baseline3 的 ckpt 格式不同，不能混用。
set -u
cd "$(dirname "$0")"

SRC=baseline1
OUT=/tmp/cxr_t4
SUB=${1:-$SRC/submission_ens.csv}

echo "== 1/5 检查素材 =="
for f in "$SRC/runs/full_base/best.pth" "$SRC/runs/full_effv2m/best.pth" \
         "$SRC/pictures_100" "$SRC/pictures_5" "$SUB"; do
  if [ ! -e "$f" ]; then echo "✗ 缺: $f"; exit 1; fi
done
echo "  ✓ 权重 / 图片 / 提交文件齐全"

rm -rf "$OUT"
mkdir -p "$OUT/weights" "$OUT/pictures_100" "$OUT/pictures_5"

echo "== 2/5 复制代码（排除 runs / data600 / 权重 / 缓存）=="
rsync -a --exclude 'runs' --exclude 'data600' --exclude '__pycache__' \
  --exclude 'pictures_*' --exclude '*.pth' --exclude '*.log' \
  "$SRC/" "$OUT/baseline1/"

echo "== 3/5 复制权重与图片 =="
cp "$SRC/runs/full_base/best.pth"   "$OUT/weights/full_base.pth"
cp "$SRC/runs/full_effv2m/best.pth" "$OUT/weights/full_effv2m.pth"
cp -r "$SRC/pictures_100/." "$OUT/pictures_100/"
cp -r "$SRC/pictures_5/."   "$OUT/pictures_5/"
cp "$SUB" "$OUT/submission_ens.csv"

echo "== 4/5 写 README（Kaggle 上照抄即可）=="
cat > "$OUT/README_T4.md" <<'EOF'
# T4 核验包

## Kaggle 里怎么跑

```python
!cp -r /kaggle/input/cxr-deliver/* /kaggle/working/
%cd /kaggle/working/baseline1
!python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
!python cost.py ../submission_ens.csv ../pictures_100 ../pictures_5 \
    --ckpt ../weights/full_base.pth ../weights/full_effv2m.pth
```

## 只看两行

```
最终平均耗时 XXX ms，未超过上限 / 超过上限
不一致 N 张
```

- ≤100ms 且不一致 0 张 → 77.67 可交付，直接定稿；
- 100~150ms → 还有一次整改复测机会（只许优化推理实现）；
- >150ms → 448 双模型不可行，必须换单模型或降分辨率。

## 注意

- Kaggle 的 CPU 通常弱于官方容器 → 这里的数字偏大，属保守判定；
- 必须在 GPU T4 环境下跑（不要用 CPU 加速器）；
- 权重是 fp32；若最终要用 fp16 交付，需重测一次（推理精度会写在 cost.py 输出里）。
EOF

echo "== 5/5 打包 =="
tar czf /tmp/cxr_t4.tgz -C /tmp cxr_t4
ls -lh /tmp/cxr_t4.tgz
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum /tmp/cxr_t4.tgz
else
  shasum -a 256 /tmp/cxr_t4.tgz
fi

cat <<'EOT'

下一步：
  1) 下载到本机： scp lcy@<服务器>:...  # 或直接在这台机器上操作
  2) 上传 Kaggle：Datasets → New Dataset → 拖入解压后的 cxr_deliver 目录
  3) Notebook 里照 README_T4.md 的两行命令跑
EOT
