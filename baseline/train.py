# -*- coding: utf-8 -*-
"""训练主脚本（从 Kaggle 两个训练 notebook 的流程整合而来）。

用法：
    python train.py --model evax                 # 训练 EVA-X（6 epoch）
    python train.py --model chexfound            # 训练 CheXFound（6 epoch）
    python train.py --model evax --data train+valid   # 联合训练集+验证集出正式模型
    python train.py --model evax --epochs 3 --batch-size 8   # 快速试跑

每个 epoch 保存两个权重文件到 runs/<model>_<时间戳>/：
    model_f{epoch}.pth   普通权重
    ema_f{epoch}.pth     EMA 权重（推理默认用这个）
另外把验证集上 study 级 Macro AUC 最好的 EMA 权重另存为 best.pth。

验证指标同时打印「图像级」和「study 级」两组 Macro AUC —— 比赛的评分单位是
Study_id（一次检查），所以 study 级那组才是选型依据。
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    aggregate_by_study,
    encode_labels,
    format_auc_report,
    macro_auc,
    per_class_auc,
    read_split_csv,
)
from dataset import ChestXrayDataset, make_train_tfms, make_val_tfms
from ema import ModelEMA, get_model
from losses import WeightedFocalBCELoss, calculate_pos_weights
from models import build_model, build_optimizer, default_batch_size, model_img_size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cosine_scheduler(optimizer, total_steps, warmup_ratio):
    """cosine 学习率 + 线性 warmup（按 step 更新）。"""
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, ema, scaler,
                    device, use_amp):
    """单个 epoch 的训练（AMP 混合精度 + EMA 更新）。"""
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="[Train]")
    for imgs, _ids, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        running += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, use_amp, val_rows, agg="mean"):
    """在验证集上推理，返回 (image_auc, study_auc, per_class_auc_study)。"""
    model.eval()
    all_probs, all_labels = [], []
    for imgs, _ids, labels in tqdm(loader, desc="[Val]"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(imgs)
        all_probs.append(outputs.sigmoid().float().cpu().numpy())
        all_labels.append(labels.numpy())

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)

    # 图像级（参考）
    img_auc = macro_auc(labels, probs)
    # study 级（比赛真正的评分单位）
    _, study_probs, study_labels = aggregate_by_study(val_rows, probs,
                                                      labels=labels, how=agg)
    study_auc = macro_auc(study_labels, study_probs)
    cls_auc = per_class_auc(study_labels, study_probs)
    return img_auc, study_auc, cls_auc


def main():
    parser = argparse.ArgumentParser(description="胸部 X 光多标签分类训练")
    parser.add_argument("--model", default="evax", choices=["evax", "chexfound"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data", default="train",
                        help="train=只用训练集(验证集监控) | train+valid=联合训练")
    parser.add_argument("--agg", default="mean", choices=["mean", "max"],
                        help="study 级聚合方式")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="输出目录（默认自动生成）")
    parser.add_argument("--resume", default=None, help="从某个 ckpt 恢复训练")
    parser.add_argument("--limit", type=int, default=0,
                        help="只取前 N 张训练图做冒烟测试（0 = 全部；"
                             "验证集同步截到 256 张，Mac 无 GPU 时用它验证流程）")
    args = parser.parse_args()

    from config import (
        EPOCHS, EMA_DECAY, FOCAL_ALPHA, FOCAL_BCE_MIX, FOCAL_GAMMA,
        NUM_WORKERS, POS_WEIGHT_BETA, POS_WEIGHT_CAP, RUNS_DIR, SEED,
        USE_AMP, WARMUP_RATIO, WEIGHT_DECAY,
    )

    epochs = args.epochs or EPOCHS
    batch_size = args.batch_size or default_batch_size(args.model)
    seed = args.seed if args.seed is not None else SEED
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"设备: {device} | AMP: {use_amp} | batch: {batch_size} | "
          f"epochs: {epochs} | seed: {seed}")

    # ---------- 数据 ----------
    train_rows = read_split_csv("train")
    if args.data == "train+valid":
        train_rows = train_rows + read_split_csv("valid")
        val_rows = None
    else:
        val_rows = read_split_csv("valid")
    if args.limit:
        # 冒烟测试模式：训练/验证都截小，让 Mac（无 GPU）也能快速验证全流程
        train_rows = train_rows[:args.limit]
        if val_rows is not None:
            val_rows = val_rows[:min(256, len(val_rows))]
        print(f"[冒烟测试] 只用 {len(train_rows)} 张训练图"
              + (f" + {len(val_rows)} 张验证图" if val_rows is not None else ""))
    img_size = model_img_size(args.model)

    train_ds = ChestXrayDataset(train_rows, "train",
                                transform=make_train_tfms(img_size))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=False, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = None
    if val_rows is not None:
        val_ds = ChestXrayDataset(val_rows, "valid",
                                  transform=make_val_tfms(img_size))
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
            persistent_workers=NUM_WORKERS > 0,
        )

    # ---------- 模型 / 损失 / 优化器 ----------
    model = build_model(args.model).to(device)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 块 GPU（DataParallel）")
        model = torch.nn.DataParallel(model)

    train_labels = encode_labels(train_rows)
    pos_weights = calculate_pos_weights(train_labels, beta=POS_WEIGHT_BETA,
                                        cap=POS_WEIGHT_CAP).to(device)
    print("各类别 pos_weight:", [f"{w:.2f}" for w in pos_weights.tolist()])
    criterion = WeightedFocalBCELoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                                     pos_weights=pos_weights,
                                     mix=FOCAL_BCE_MIX)
    optimizer = build_optimizer(args.model, model,
                                weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    scheduler = make_cosine_scheduler(optimizer, total_steps, WARMUP_RATIO)
    # 纯 CPU 调试时 use_amp=False，scaler 用 cpu 设备避免无 CUDA 环境报错
    scaler = torch.amp.GradScaler(device="cuda" if use_amp else "cpu",
                                  enabled=use_amp)
    ema = ModelEMA(model, decay=EMA_DECAY, device=device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        get_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"从 {args.resume} 恢复，起始 epoch = {start_epoch}")

    # ---------- 输出目录 ----------
    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join(
            RUNS_DIR, f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # ---------- 训练循环 ----------
    metrics_log = []
    best_study_auc = -1.0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               criterion, ema, scaler, device, use_amp)
        log = {"epoch": epoch + 1, "train_loss": round(loss, 5)}

        if val_loader is not None:
            # 用 EMA 权重做验证（推理时用的就是它）
            img_auc, study_auc, cls_auc = validate(
                ema.ema, val_loader, device, use_amp, val_rows, args.agg
            )
            log.update({"image_auc": round(img_auc, 5),
                        "study_auc": round(study_auc, 5)})
            print(f"Epoch {epoch + 1}/{epochs} | loss {loss:.4f} | "
                  f"图像级 AUC {img_auc:.4f} | study 级 AUC {study_auc:.4f} | "
                  f"{time.time() - t0:.0f}s")
            print("  每类 AUC:", format_auc_report(cls_auc))
            if study_auc > best_study_auc:
                best_study_auc = study_auc
                torch.save({"epoch": epoch + 1,
                            "ema_state_dict": ema.state_dict()},
                           os.path.join(out_dir, "best.pth"))
                print(f"  ** 新的最佳 study AUC，已保存 best.pth **")
        else:
            print(f"Epoch {epoch + 1}/{epochs} | loss {loss:.4f} | "
                  f"{time.time() - t0:.0f}s")

        # 每 epoch 都存权重（Kaggle 方案最后平均第 5、6 两个 epoch）
        torch.save({"epoch": epoch + 1,
                    "model_state_dict": get_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": loss},
                   os.path.join(out_dir, f"model_f{epoch + 1}.pth"))
        torch.save({"epoch": epoch + 1, "ema_state_dict": ema.state_dict()},
                   os.path.join(out_dir, f"ema_f{epoch + 1}.pth"))

        metrics_log.append(log)
        with open(os.path.join(out_dir, "metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics_log, f, ensure_ascii=False, indent=2)

    print(f"训练完成，权重在: {out_dir}")
    if val_loader is not None:
        print(f"最佳 study 级 Macro AUC: {best_study_auc:.4f}")


if __name__ == "__main__":
    main()
