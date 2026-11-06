# -*- coding: utf-8 -*-
"""有监督微调主循环（ImageNet 预训练骨干 → 10 类多标签）。

用法示例：
    python train.py                                  # 默认 convnext_base @384，12 epoch
    python train.py --backbone convnext_small --epochs 15
    python train.py --ssl runs/ssl/ssl_backbone.pth   # 以自监督预训练结果为起点（推荐）
    python train.py --data train+valid                # 把验证集并入训练（规则第 3 条，见下方警告）
    python train.py --limit 64 --epochs 1             # 冒烟测试

关于验证集（规则第 3 条）：`--data train+valid` 会把验证集并进训练集。一旦并入，
就**不能再拿它做早停/选 ckpt/调阈值**。所以正确顺序是：先在 train 上定好所有超参和
ckpt 选择策略 → 最后再用 train+valid 重训出正式模型 → 并在申报材料里注明。
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import (aggregate_by_study, encode_labels, format_auc_report,
                    macro_auc, per_class_auc, read_split_csv)
from config import (BATCH_SIZE, EMA_DECAY, EPOCHS, FOCAL_ALPHA, FOCAL_GAMMA,
                    FOCAL_MIX, FREEZE_EPOCHS, GRAD_CLIP, IMG_SIZE,
                    NUM_WORKERS, POS_WEIGHT_BETA, POS_WEIGHT_CAP, RUNS_DIR,
                    USE_AMP, WARMUP_RATIO)
from dataset import ChestXrayDataset, make_train_tfms, make_eval_tfms
from ema import ModelEMA
from losses import AsymmetricLoss, WeightedFocalBCELoss, calculate_pos_weights
from models import (BACKBONE_CHOICES, build_model, build_optimizer, get_model,
                    load_ssl_backbone, set_backbone_trainable)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, total_steps, warmup_ratio=WARMUP_RATIO,
                   min_lr_ratio=0.01):
    """线性预热 + cosine 衰减（按 step 更新）。"""
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, use_amp, epoch, grad_clip, ema=None):
    model.train()
    total_loss, n = 0.0, 0
    # 进度条的 949/949 是「步数（batch 数）」，每步喂 batch_size 张图，
    # 一轮下来就把整个训练集完整过一遍（30344 张 / 32 = 948 步左右）。
    pbar = tqdm(loader, desc=f"[Train] epoch {epoch + 1}", leave=False,
                dynamic_ncols=True)
    for imgs, _dicom, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # ★ EMA 必须「每步」更新：decay=0.9997 的语义是"每步混入 0.03%"。
        #   放在 epoch 循环里只调 15 次 → EMA 几乎冻结在初始权重，
        #   验证集评估的其实是训练前的模型（2026-10-30 踩过，AUC 卡 0.58）。
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss / max(1, n):.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(1, n)


@torch.no_grad()
def validate(model, loader, device, use_amp, val_rows, agg="mean"):
    """在验证集上算图像级 / study 级 Macro AUC（study 级是选型依据）。"""
    model.eval()
    probs, labels = [], []
    for imgs, _dicom, lab in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(lab.numpy())
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    img_auc = macro_auc(labels, probs)
    _, study_probs, study_labels = aggregate_by_study(
        val_rows, probs, how=agg, labels=labels)
    study_auc = macro_auc(study_labels, study_probs)
    return img_auc, study_auc, per_class_auc(study_labels, study_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None, choices=list(BACKBONE_CHOICES),
                        help="torchvision 骨干（默认取 config.BACKBONE）")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None, help="骨干学习率")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--data", default="train", choices=["train", "train+valid"],
                        help="train+valid：把验证集并入训练（规则第 3 条，见文件头说明）")
    parser.add_argument("--ssl", default=None, help="自监督预训练得到的骨干权重")
    parser.add_argument("--freeze-epochs", type=int, default=FREEZE_EPOCHS,
                        help="前 N 轮冻结骨干，只训分类头")
    parser.add_argument("--limit", type=int, default=0, help="冒烟测试：只用前 N 条")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--pos-cap", type=float, default=POS_WEIGHT_CAP,
                        help="pos_weight 上限（默认 10）。罕见类是最弱环节，"
                             "调高到 20 相当于专攻类1/3/4，是最便宜的旋钮")
    parser.add_argument("--pos-beta", type=float, default=POS_WEIGHT_BETA,
                        help="effective-number 的 beta（默认 0.9999）")
    parser.add_argument("--balanced-sampler", type=float, default=0.0,
                        help="类别平衡采样强度 alpha（0=关闭，推荐 0.5）。"
                             "用 (N/n_c)^alpha 给含罕见类的图加权后按权重有放回采样。"
                             "它与 pos_weight 解决的是不同问题：pos_weight 改惩罚力度，"
                             "采样改出现频次；类1 的病根是样本量，所以采样才是对症的。")
    parser.add_argument("--balanced-cap", type=float, default=4.0,
                        help="单张图的最大采样权重（默认 4.0，防止罕见类把常见类挤没）")
    parser.add_argument("--loss", default="focal", choices=["focal", "asl"],
                        help="损失函数：focal=原来的 Focal+加权BCE；"
                             "asl=Asymmetric Loss（负样本大幅降权、正样本不降权），"
                             "专治罕见类信号被海量负样本淹没")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0,
                        help="ASL 负样本的聚焦强度（默认 4.0；调小=更保守）")
    parser.add_argument("--asl-clip", type=float, default=0.05,
                        help="ASL 负样本概率平移量（默认 0.05；0=不平移）")
    parser.add_argument("--train-csv", default=None,
                        help="自定义训练 CSV（患者级 K 折用，见 make_folds.py）")
    parser.add_argument("--val-csv", default=None,
                        help="自定义验证 CSV（K 折用）；给了空路径/文件不存在则不做验证")
    parser.add_argument("--out", default=None, help="输出目录")
    parser.add_argument("--resume", default=None, help="从 ckpt 恢复（会重置调度器）")
    args = parser.parse_args()

    from config import BACKBONE, LR
    backbone = args.backbone or BACKBONE
    lr = args.lr or LR

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    print(f"设备: {device} | AMP: {use_amp} | 骨干: {backbone} | "
          f"batch: {args.batch_size} | epochs: {args.epochs} | seed: {args.seed}")
    if args.data == "train+valid":
        print("⚠ 已启用 train+valid：训练集已并入验证集，本轮结果不能再用于"
              "早停/选 ckpt/调参（规则第 3 条），且须在申报材料中注明。")

    # ---------- 数据 ----------
    # 训练图片恒在 data600/train 下（官方 train 与 K 折的「训练折」都来自这里）；
    # 验证图片：官方 split 在 data600/valid 下，K 折的「验证折」仍在 data600/train 下。
    train_split = "train"
    val_split = "train" if args.train_csv else "valid"
    if args.train_csv:
        from common import read_csv_path
        train_rows = read_csv_path(args.train_csv)
        print(f"训练集来自自定义 CSV: {args.train_csv}")
    elif args.data == "train+valid":
        train_rows = read_split_csv("train") + read_split_csv("valid")
    else:
        train_rows = read_split_csv("train")

    if args.val_csv is not None:
        from common import read_csv_path
        if os.path.exists(args.val_csv):
            val_rows = read_csv_path(args.val_csv)
            print(f"验证集来自自定义 CSV: {args.val_csv}")
        else:
            val_rows = None
            print(f"验证 CSV 不存在（{args.val_csv}），本轮不做验证")
    elif args.data == "train+valid":
        val_rows = None
    else:
        val_rows = read_split_csv("valid")
    if args.limit:
        train_rows = train_rows[:args.limit]
        print(f"[冒烟测试] 只用 {len(train_rows)} 张训练图")
    if val_rows is not None and args.limit:
        val_rows = val_rows[:max(256, args.limit * 4)]
        print(f"[冒烟测试] 验证集截断到 {len(val_rows)} 张")

    train_ds = ChestXrayDataset(train_rows, train_split,
                                transform=make_train_tfms(args.img_size))
    train_labels = encode_labels(train_rows)

    # ---------- 类别平衡采样（专攻罕见类，尤其是类1） ----------
    # 病根：类1 只有 894 个正样本（2.95%），而 Macro AUC 十类等权，它一个人就能压掉 0.008。
    # pos_weight 只改"错判的惩罚"，不改变"见过的次数"；采样才真正改变样本量——所以它和权重是两件事。
    sampler = None
    if args.balanced_sampler > 0:
        freq = train_labels.sum(axis=0).astype(np.float64)          # 每类正样本数
        cls_w = (len(train_labels) / np.maximum(freq, 1.0)) ** args.balanced_sampler
        w = np.ones(len(train_labels), dtype=np.float64)
        for c in range(train_labels.shape[1]):
            w += train_labels[:, c] * (cls_w[c] - 1.0)               # 含罕见类的图 → 权重更高
        w = np.clip(w, 1.0, args.balanced_cap)
        sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w),
                                        replacement=True)
        print(f"[平衡采样] alpha={args.balanced_sampler} cap={args.balanced_cap} | "
              f"每类权重 {[f'{x:.1f}' for x in cls_w.tolist()]}")
        print(f"[平衡采样] 图片权重 min {w.min():.2f} / mean {w.mean():.2f} / max {w.max():.2f}"
              f"（类1 的图约被多采 {min(cls_w[0], args.balanced_cap):.1f} 倍）")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=len(train_ds) > args.batch_size, persistent_workers=args.workers > 0,
    )
    # 把「步数 vs 张数」说清楚，避免把 949 步误读成 949 张
    print(f"训练集 {len(train_ds)} 张 | batch {args.batch_size} | "
          f"每轮 {len(train_loader)} 步 × {args.batch_size} 张 "
          f"≈ {len(train_loader) * args.batch_size} 张次"
          f"（每轮都完整过一遍训练集）")
    if val_rows is not None:
        print(f"验证集 {len(val_rows)} 张（{len({r['Study_id'] for r in val_rows})} "
              f"个检查），每轮结束后评估")
    val_loader = None
    if val_rows is not None:
        val_ds = ChestXrayDataset(val_rows, val_split,
                                  transform=make_eval_tfms(args.img_size))
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.workers, pin_memory=(device.type == "cuda"),
            persistent_workers=args.workers > 0,
        )

    # ---------- 模型 / 损失 / 优化器 ----------
    model = build_model(backbone, pretrained=True)
    if args.ssl:
        model = load_ssl_backbone(model, args.ssl)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 块 GPU（DataParallel）")
        model = torch.nn.DataParallel(model)

    # train_labels 已在构建 DataLoader 之前算好（平衡采样要用）
    # pos_weight 上限可调：Macro AUC 十类等权，罕见类（类1 仅 894 正样本）是最弱的一环，
    # 按 effective-number 公式类1 应得约 32.9，但默认封顶 10。把 cap 调高是"专攻罕见类"
    # 最便宜的一个旋钮（零代码改动，只换一个数），值得单独做一组消融。
    pos_weights = calculate_pos_weights(train_labels, beta=args.pos_beta,
                                        cap=args.pos_cap).to(device)
    print(f"各类别 pos_weight（beta={args.pos_beta}, cap={args.pos_cap}）:",
          [f"{w:.2f}" for w in pos_weights.tolist()])
    print(f"EMA: 每训练步更新一次 | decay {EMA_DECAY}（带 warmup）"
          f" | 每轮约 {len(train_loader)} 次更新")
    if args.loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=0.0,
                                   clip=args.asl_clip)
        print(f"损失: ASL（gamma_neg={args.asl_gamma_neg}, clip={args.asl_clip}）"
              f"—— 负样本大幅降权，正样本不降权；此时 pos_weight 不再参与")
    else:
        criterion = WeightedFocalBCELoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                                         pos_weights=pos_weights, mix=FOCAL_MIX)
        print(f"损失: Focal×{FOCAL_MIX} + 加权BCE×{1 - FOCAL_MIX:.1f}")
    optimizer = build_optimizer(get_model(model), lr=lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps)
    scaler = torch.amp.GradScaler(device="cuda" if use_amp else "cpu",
                                  enabled=use_amp)
    ema = ModelEMA(model, decay=EMA_DECAY, device=device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        get_model(model).load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        # 把学习率曲线「快进」到中断位置：否则恢复后 LR 会从预热重新爬升，
        # 相当于给已经收敛的模型一次冲击（这个坑之前在 EVA-X 上踩过）。
        skip_steps = start_epoch * len(train_loader)
        for _ in range(skip_steps):
            scheduler.step()
        print(f"从 {args.resume} 恢复：已完成 {start_epoch} 轮，"
              f"学习率曲线快进 {skip_steps} 步（优化器动量重置，属热重启）")

    out_dir = args.out or os.path.join(
        RUNS_DIR, f"{backbone}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # ---------- 训练 ----------
    metrics_log = []
    best_auc = -1.0
    for epoch in range(start_epoch, args.epochs):
        if args.freeze_epochs and epoch < args.freeze_epochs:
            if epoch == 0:
                print(f"[冻结] 前 {args.freeze_epochs} 轮只训分类头")
            set_backbone_trainable(model, False)
        elif args.freeze_epochs and epoch == args.freeze_epochs:
            print("[解冻] 骨干参数开始参与训练")
            set_backbone_trainable(model, True)

        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer,
                               scheduler, scaler, device, use_amp, epoch,
                               GRAD_CLIP, ema=ema)
        log = {"epoch": epoch + 1, "train_loss": round(loss, 5)}

        if val_loader is not None:
            img_auc, study_auc, cls_auc = validate(
                ema.ema, val_loader, device, use_amp, val_rows)
            log.update({"image_auc": round(img_auc, 5),
                        "study_auc": round(study_auc, 5)})
            print(f"Epoch {epoch + 1}/{args.epochs} | loss {loss:.4f} | "
                  f"图像级 AUC {img_auc:.4f} | study 级 AUC {study_auc:.4f} | "
                  f"{time.time() - t0:.0f}s | EMA 更新 {ema.num_updates} 步")
            print("  每类 AUC:", format_auc_report(cls_auc))
            if study_auc > best_auc:
                best_auc = study_auc
                torch.save({
                    "model_state_dict": ema.ema.state_dict(),
                    "epoch": epoch + 1, "backbone": backbone,
                    "img_size": args.img_size,
                    # 训练用的源图边长（data600 → 600，data1024 → 1024）。
                    # 推理端靠这个值决定 draft 解码尺度，必须与训练一致。
                    "src_size": int(os.environ.get("CXR_SRC_SIZE", 600)),
                    "num_classes": get_model(model).num_classes,
                    "study_auc": study_auc,
                }, os.path.join(out_dir, "best.pth"))
                print("  ** 新的最佳 study AUC，已保存 best.pth **")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs} | loss {loss:.4f} | "
                  f"{time.time() - t0:.0f}s（未启用验证集）")

        # 每轮都存 EMA 权重（推理用最后两个 epoch 平均）
        torch.save({
            "model_state_dict": ema.ema.state_dict(),
            "epoch": epoch + 1, "backbone": backbone,
            "img_size": args.img_size,
            "src_size": int(os.environ.get("CXR_SRC_SIZE", 600)),
            "num_classes": get_model(model).num_classes,
        }, os.path.join(out_dir, f"ema_f{epoch + 1}.pth"))

        metrics_log.append(log)
        with open(os.path.join(out_dir, "metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics_log, f, ensure_ascii=False, indent=2)

    print(f"训练完成，权重在: {out_dir}")
    if val_loader is not None:
        print(f"最佳 study 级 Macro AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
