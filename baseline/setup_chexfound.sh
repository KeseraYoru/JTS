#!/usr/bin/env bash
# 准备 CheXFound 的第三方代码，并检查权重是否就位。
#
# 现实流程（国内网络）：
#   1. 代码仓库：git clone（GitHub 国内一般可达；不行就手动下载 zip 解压）
#   2. 权重 4 件套：Google Drive 在国内无法直接下，需要在能访问的网络/设备上
#      手动下载，再传到 third_party/CheXFound_Model/（本脚本只负责检查在不在）
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p third_party
cd third_party

# ---------- 1) 克隆官方代码仓库 ----------
if [ ! -d CheXFound ]; then
    echo "==> git clone RPIDIAL/CheXFound"
    if ! git clone --recursive https://github.com/RPIDIAL/CheXFound.git; then
        echo "!! clone 失败（服务器可能不通 GitHub）。改用本地方案：" >&2
        echo "   在 Mac 上执行: git clone --recursive https://github.com/RPIDIAL/CheXFound.git" >&2
        echo "   然后传到服务器: rsync -av --partial --progress ~/CheXFound/ 用户名@服务器:~/cxr/baseline/third_party/CheXFound/" >&2
        exit 1
    fi
else
    echo "==> CheXFound 仓库已存在，跳过 clone"
fi

# ---------- 2) 检查 4 个权重/配置文件（人工下载放入） ----------
mkdir -p CheXFound_Model
missing=0
for f in config.yaml teacher_checkpoint.pth glori.pth results_eval_linear.json; do
    if [ -f "CheXFound_Model/$f" ]; then
        echo "==> $f 就位（$(du -h "CheXFound_Model/$f" | cut -f1)）"
    else
        echo "!! 缺少 $f" >&2
        missing=1
    fi
done

if [ "$missing" = "1" ]; then
    echo "" >&2
    echo "这 4 个文件请从 Google Drive 官方文件夹手动下载：" >&2
    echo "  https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806" >&2
    echo "（国内打不开就换能访问谷歌的网络/设备下载，浏览器直接下，不要用脚本）" >&2
    echo "下载后放到: $(pwd)/CheXFound_Model/" >&2
    echo "参考大小: config.yaml 约 3KB | teacher_checkpoint.pth 约 1.5G | glori.pth 约 900M | json 约 1.4KB" >&2
    exit 1
fi

# ---------- 3) 引导后续步骤 ----------
echo ""
echo "==> 代码和权重就位。接下来三步："
echo "   1. 装依赖:   pip install fvcore open_clip_torch omegaconf scikit-multilearn torchmetrics iopath pandas transformers scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple"
echo "      （xformers/mmcv 等官方完整环境组件不用装，代码会自动回退原生 attention）"
echo "   2. 一键检查: python3 check_chexfound.py"
echo "   3. 全绿了再训: python3 train.py --model chexfound --epochs 6"
