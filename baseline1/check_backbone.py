# -*- coding: utf-8 -*-
"""骨干特征诊断：判断「预训练权重（含自监督）是否已经坏了」。

背景：如果自监督把骨干训崩（特征坍缩 / 破坏掉 ImageNet 特征），微调时会出现
「loss 一直降、但验证 AUC 卡在 0.5x」的现象——因为模型只能记住训练集的噪声，
学不到可泛化的特征。

用法：
    python check_backbone.py                                  # 只看 ImageNet 初始特征
    python check_backbone.py --ssl runs/ssl/ssl_backbone.pth  # 对比 ImageNet vs SSL

判读标准（在 16 张验证图上提取特征）：
    健康：跨样本 std 明显 >0（通常 0.2~0.8）、样本间余弦相似度 0.2~0.8
    坍缩：跨样本 std ≈ 0、样本间余弦相似度 ≈ 1.0（所有图特征几乎一样 → 骨干废了）
"""
import argparse

import numpy as np
import torch
from PIL import Image

from common import prep_img_path, read_split_csv
from config import BACKBONE, IMG_SIZE
from dataset import make_eval_tfms
from models import build_model


def extract(model, paths, img_size, device):
    """跑 backbone（不含分类头），返回 [N, D] 特征。"""
    tfm = make_eval_tfms(img_size)
    feats = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            with Image.open(p) as im:
                x = tfm(im.convert("RGB")).unsqueeze(0).to(device)
            feats.append(model.backbone(x).float().cpu().numpy()[0])
    return np.stack(feats)


def report(name, F):
    """打印坍缩指标。"""
    std = float(F.std(axis=0).mean())
    scale = float(np.abs(F).mean()) + 1e-8
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    off = S[~np.eye(len(F), dtype=bool)]
    rel_std = std / scale
    print(f"{name}")
    print(f"  特征维度 {F.shape[1]} | 跨样本 std {std:.4f}（相对 {rel_std:.3f}）"
          f" | 特征绝对值均值 {scale:.3f}")
    print(f"  样本间余弦相似度：均值 {off.mean():.3f}，最大 {off.max():.3f}"
          f"（越接近 1 越像坍缩）")
    if rel_std < 0.02 or off.mean() > 0.98:
        print("  ⚠ 判定：特征坍缩/退化——这个骨干基本没有区分能力")
    else:
        print("  ✓ 判定：特征有区分度")
    return off.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", default=None, help="自监督权重（可选，用于对比）")
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--n", type=int, default=16, help="用多少张验证图")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_split_csv("valid")[:args.n]
    paths = [prep_img_path(r, "valid") for r in rows]
    print(f"设备 {device} | 骨干 {args.backbone} | 输入 {args.img_size}px | "
          f"样本 {len(paths)} 张\n")

    # ① ImageNet 初始权重
    model = build_model(args.backbone, pretrained=True).to(device)
    F_imgnet = extract(model, paths, args.img_size, device)
    report("【ImageNet 初始权重】", F_imgnet)

    # ② 自监督权重（如果给了）
    if args.ssl:
        print("")
        from models import load_ssl_backbone
        model2 = build_model(args.backbone, pretrained=True)
        model2 = load_ssl_backbone(model2, args.ssl).to(device)
        F_ssl = extract(model2, paths, args.img_size, device)
        report("【自监督后的骨干】", F_ssl)

        # ③ 两组特征的差异（相对变化）
        rel = float(np.linalg.norm(F_ssl - F_imgnet) /
                    (np.linalg.norm(F_imgnet) + 1e-8))
        print("")
        print(f"SSL 特征相对 ImageNet 的变化幅度：{rel:.3f}")
        print("  0.1~0.4 = 温和调整（正常）；>0.8 = 改动很大（有破坏风险）")


if __name__ == "__main__":
    main()
