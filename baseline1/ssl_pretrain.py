# -*- coding: utf-8 -*-
"""自监督预训练（SimCLR）：在**官方训练集**上做无标签预训练，再做有监督微调。

为什么允许（规则补充四，原文要点）：
「在不引入任何外部医学数据和测试集的前提下，使用比赛官方提供的训练集图像进行
自监督预训练，属于《规则》第 1 条允许的范围。」
并明确：禁止的是「外部或非官方医学数据」——官方训练集是赛方提供的数据，不在此列。

所以这条路线完全合规，而且能解决一个真实问题：ImageNet 是自然照片，与 X 光片
的灰度、对比度、纹理差异很大；在官方数据上自监督一遍，相当于让骨干先「熟悉这个域」，
再去做有监督微调，通常能比直接微调高 1~3 个点。

用法：
    python ssl_pretrain.py                      # 默认 convnext_base @384，20 epoch
    python ssl_pretrain.py --epochs 30 --batch-size 96
    python ssl_pretrain.py --limit 2000         # 冒烟测试

产出：runs/ssl/ssl_backbone.pth（只有骨干权重），供 train.py --ssl 使用。
申报：自监督只用官方 train split，须在「模型核验材料与数据声明清单.xlsx」中说明。
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import read_split_csv
from config import (BACKBONE, IMG_SIZE, NUM_CLASSES, RUNS_DIR, SSL_BASE_LR,
                    SSL_BATCH_SIZE, SSL_EPOCHS, SSL_LR, SSL_OUT,
                    SSL_TEMPERATURE, USE_AMP)
from dataset import SSLTwoView
from models import BACKBONE_CHOICES, build_model


class SimCLRModel(nn.Module):
    """编码器（复用 ChestXrayModel 的 backbone）+ 投影头。"""

    def __init__(self, backbone_name, out_dim=256):
        super().__init__()
        base = build_model(backbone_name, num_classes=NUM_CLASSES,
                           pretrained=True)
        self.encoder = base.backbone
        feat_dim = base.head[-1].in_features
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        return F.normalize(self.projector(h), dim=-1)


def nt_xent(z1, z2, temperature):
    """NT-Xent：同一张图的两个视图互为正样本，批内其余全为负样本。"""
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)              # [2B, D]
    sim = (z @ z.t()) / temperature             # [2B, 2B]
    # 屏蔽对角线（自己和自己）
    mask = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))
    # 每个样本的正样本：另半边对应位置
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=BACKBONE, choices=list(BACKBONE_CHOICES))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=SSL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=SSL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=SSL_BASE_LR)
    parser.add_argument("--temp", type=float, default=SSL_TEMPERATURE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=SSL_OUT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    print(f"设备 {device} | 骨干 {args.backbone} | batch {args.batch_size} | "
          f"epochs {args.epochs}")
    print("数据范围：仅官方 train split（规则补充四允许的自监督范围）")

    rows = read_split_csv("train")          # 只用官方训练集，不碰 valid/test
    if args.limit:
        rows = rows[:args.limit]
        print(f"[冒烟测试] 只用 {len(rows)} 张")
    ds = SSLTwoView(rows, "train", args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.workers > 0)

    model = SimCLRModel(args.backbone).to(device)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 块 GPU（DataParallel）")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    total_steps = max(1, args.epochs * len(loader))
    warmup = max(1, int(total_steps * 0.05))

    def lr_fn(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    scaler = torch.amp.GradScaler(device="cuda" if use_amp else "cpu",
                                  enabled=use_amp)

    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for v1, v2 in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent(z1.float(), z2.float(), args.temp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * v1.size(0)
            n += v1.size(0)
        print(f"SSL Epoch {epoch + 1}/{args.epochs} | loss {total / max(1, n):.4f} "
              f"| {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    inner = model.module if hasattr(model, "module") else model
    torch.save({
        "backbone": inner.encoder.state_dict(),
        "backbone_name": args.backbone,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "data": "official train split only",
        "method": "SimCLR",
    }, args.out)
    print(f"自监督权重已保存: {args.out}")
    print("下一步：python train.py --ssl " + args.out)


if __name__ == "__main__":
    main()
