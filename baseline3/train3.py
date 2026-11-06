# -*- coding: utf-8 -*-
"""训练主脚本：换预训练起点 + 把 baseline1 的坑全部前置规避。

用法速查
--------
    # 0) 自检（不训练，只验证架构与权重要不要下载、能不能对上）
    python train3.py --arch dinov2_vitb14 --selftest

    # 1) 探针：2 轮、冻结 1 轮，只回答"这个起点好不好"（约 15 分钟）
    python train3.py --arch dinov2_vitb14 --img-size 384 --batch-size 16 --probe \
        --train-csv runs/folds/fold_0_train.csv --val-csv runs/folds/fold_0_val.csv \
        --out runs/probe_dino

    # 2) 全量微调
    python train3.py --arch dinov2_vitb14 --img-size 384 --batch-size 16 --epochs 12 \
        --out runs/full_dino

关键设计（每条都对应一次踩过的坑）
--------------------------------
1. LR 按 batch 线性缩放：`--lr auto` 时 lr = 3e-4 × batch/32。baseline1 在 batch8 上
   沿用 3e-4，实测掉 0.0067，一度被误判成"batch 的错"。
2. EMA 每 step 更新，并打印累计步数；结束前核对 = 轮数 × 每轮步数。
3. ckpt 里写全 `impl / arch / img_size / src_size / view_dim / pool`：
   推理端据此重建结构，避免"权重对不上结构"这种只有到预测时才炸的问题。
4. 训练与推理的分辨率必须一致：`--img-size` 直接写进 ckpt，predict 从 ckpt 取值。
5. 冻结骨干时先解包 DataParallel（baseline1 在这里崩过）。
6. 默认不启用 ASL / 平衡采样（两者在 batch8 混淆下均为负收益）。
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from archs import ALL_ARCHS, load_ssl_into_backbone, param_stats
from common3 import (aggregate_by_study, apply_normal_gating, encode_labels,
                     format_auc_report, macro_auc, per_class_auc, read_csv_path,
                     read_split_csv)
from config3 import (AGG, AUG, BATCH_SIZE, DROPOUT, EMA_DECAY, EPOCHS, FREEZE_EPOCHS,
                     FULL_BATCH, GATE_GAMMA, GRAD_CLIP, HEAD_LR_MULT, IMG_SIZE,
                     LAYER_DECAY, LR, LR_SCALE, NUM_CLASSES, NUM_WORKERS,
                     POS_WEIGHT_BETA, POS_WEIGHT_CAP, PREP_ROOT, RUNS_DIR,
                     SRC_SIZE, USE_AMP, VIEW_DIM, WARMUP_EPOCHS, WEIGHT_DECAY)
from data3 import ChestXrayDataset, make_eval_tfms, make_train_tfms, view_stats
from ema3 import ModelEMA
from losses3 import (AsymmetricLoss, WeightedFocalBCELoss, calculate_pos_weights,
                     mixup_batch)
from model3 import CXRNet, get_inner, set_backbone_trainable, trainable_param_groups


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scaled_lr(base_lr, batch_size, mode):
    if mode == "none":
        return base_lr
    if mode == "linear":
        return base_lr * batch_size / FULL_BATCH
    if mode == "sqrt":
        return base_lr * math.sqrt(batch_size / FULL_BATCH)
    raise ValueError(f"未知 LR_SCALE: {mode}")


def lr_factor(step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(max(prog, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def build_loader(dataset, batch_size, workers, shuffle, pin):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin, drop_last=False,
                      persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, loader, rows, device, use_amp, gate_gamma=0.0):
    model.eval()
    outs = []
    for imgs, _dicom, _labels, view in loader:
        imgs = imgs.to(device, non_blocking=True)
        view = view.to(device, non_blocking=True) if model.view_dim else None
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs, view)
        outs.append(torch.sigmoid(logits).float().cpu().numpy())
    probs = np.concatenate(outs, axis=0)
    labels = encode_labels(rows)
    _, sp, sl = aggregate_by_study(rows, probs, how=AGG, labels=labels)
    auc_plain = macro_auc(sl, sp)
    auc_gated = macro_auc(sl, apply_normal_gating(sp, gate_gamma)) if gate_gamma else None
    return auc_plain, auc_gated, per_class_auc(sl, sp), sp


def selftest(args):
    """只验证架构能否构建、权重能否对上，不做训练。"""
    print("=" * 70)
    print(f"[自检] arch={args.arch} | img_size={args.img_size} | "
          f"view_dim={VIEW_DIM if args.use_view else 0}")
    model = CXRNet(arch=args.arch, num_classes=NUM_CLASSES, img_size=args.img_size,
                   pretrained=args.pretrained, pool=args.pool,
                   view_dim=VIEW_DIM if args.use_view else 0, drop=DROPOUT)
    x = torch.randn(2, 3, args.img_size, args.img_size)
    out = model(x)
    st = param_stats(model)
    print(f"[自检] 前向输出 {tuple(out.shape)} | 参数量 {st['params_m']:.1f}M | "
          f"fp16 {st['fp16_mb']:.0f}MB | fp32 {st['fp32_mb']:.0f}MB")
    print(f"[自检] 合规体积（fp16）：{'✓ 在 500MB 内' if st['fp16_mb'] <= 500 else '✗ 超 500MB'}")
    print(f"[自检] 权重来源：{model.info}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="convnext_base", choices=list(ALL_ARCHS))
    p.add_argument("--pretrained", default="auto",
                   help="auto=按 arch 自动下载/使用预训练；none=随机；也可给本地 .pth 路径")
    p.add_argument("--ssl", default=None, help="自监督骨干权重（官方训练集 SSL，规则允许）")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", default="auto",
                   help="auto=按 batch 缩放；也可给具体数值（骨干学习率）")
    p.add_argument("--lr-scale", default=LR_SCALE, choices=["linear", "sqrt", "none"])
    p.add_argument("--head-lr-mult", type=float, default=HEAD_LR_MULT)
    p.add_argument("--layer-decay", type=float, default=LAYER_DECAY)
    p.add_argument("--warmup-epochs", type=float, default=WARMUP_EPOCHS)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    p.add_argument("--freeze-epochs", type=int, default=FREEZE_EPOCHS)
    p.add_argument("--probe", action="store_true",
                   help="探针模式：2 轮 + 冻结 1 轮，只用于比较起点好坏")
    p.add_argument("--loss", default="focal", choices=["focal", "asl"])
    p.add_argument("--asl-gamma-neg", type=float, default=4.0)
    p.add_argument("--pos-cap", type=float, default=POS_WEIGHT_CAP)
    p.add_argument("--aug", default=AUG, choices=["mild", "strong"])
    p.add_argument("--mixup", type=float, default=0.0, help="mixup alpha（0=关闭，建议 0.2）")
    p.add_argument("--pool", default="avg", choices=["avg", "attn"])
    p.add_argument("--use-view", action="store_true", help="拼入 ViewPosition 元数据")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--limit", type=int, default=0, help="冒烟测试：每边只用前 N 张")
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--gate-gamma", type=float, default=GATE_GAMMA)
    p.add_argument("--selftest", action="store_true", help="只做架构/权重自检")
    p.add_argument("--out", default=os.path.join(RUNS_DIR, "exp"))
    args = p.parse_args()

    if args.probe:
        args.epochs = 2
        args.freeze_epochs = max(args.freeze_epochs, 1)

    # DINOv2 的 patch 是 14：输入必须能被 14 整除（384/14=27.43 不行）。
    # 这里自动就近修正，并明确提示，避免"patch grid 与 pos_embed 对不上"这种隐蔽错误。
    if "dinov2" in args.arch and args.img_size % 14 != 0:
        fixed = int(round(args.img_size / 14)) * 14
        print(f"[尺寸] {args.arch} 要求 14 的倍数：{args.img_size} -> {fixed}"
              f"（392=28×28 token，448=32×32 token）")
        args.img_size = fixed

    setup_seed(args.seed)
    if args.selftest:
        selftest(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0

    # ---------- 数据 ----------
    if args.train_csv:
        train_rows = read_csv_path(args.train_csv)
        train_split = "train"
    else:
        train_rows = read_split_csv("train")
        train_split = "train"
    if args.val_csv:
        val_rows = read_csv_path(args.val_csv)
        val_split = "train"      # 折的图仍在 data600/train 下
    else:
        val_rows = read_split_csv("valid")
        val_split = "valid"

    if args.limit:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:args.limit * 2]
        print(f"[冒烟] 训练 {len(train_rows)} 张 / 验证 {len(val_rows)} 张（按顺序截取，结果无意义）")

    print("=" * 70)
    print(f"设备: {device} | 卡数 {n_gpu} | AMP {use_amp} | 骨干 {args.arch} | "
          f"输入 {args.img_size}px | 池化 {args.pool} | 增强 {args.aug}")
    print(f"预处理图: {PREP_ROOT} | 源边长 {SRC_SIZE}px")
    print(f"ViewPosition 分布（训练集）: {view_stats(train_rows)}")

    model = CXRNet(arch=args.arch, num_classes=NUM_CLASSES, img_size=args.img_size,
                   pretrained=args.pretrained, pool=args.pool,
                   view_dim=VIEW_DIM if args.use_view else 0, drop=DROPOUT)
    if args.ssl:
        load_ssl_into_backbone(model.backbone, args.ssl)

    st = param_stats(model)
    print(f"参数量 {st['params_m']:.1f}M | fp16 {st['fp16_mb']:.0f}MB | "
          f"fp32 {st['fp32_mb']:.0f}MB")

    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
        print(f"使用 {n_gpu} 块 GPU（DataParallel）")

    # ---------- 损失 ----------
    train_labels = encode_labels(train_rows)
    pos_w = calculate_pos_weights(train_labels, beta=POS_WEIGHT_BETA,
                                  cap=args.pos_cap).to(device)
    print(f"各类 pos_weight（cap={args.pos_cap}）: "
          f"{[f'{w:.2f}' for w in pos_w.tolist()]}")
    if args.loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg).to(device)
        print("[损失] ASL（注意：baseline1 在错误 batch/LR 下测出负收益，本次为复测）")
    else:
        criterion = WeightedFocalBCELoss(pos_weights=pos_w).to(device)

    # ---------- 数据加载 ----------
    per_gpu_bs = max(1, args.batch_size // max(1, n_gpu))
    train_ds = ChestXrayDataset(train_rows, train_split,
                                transform=make_train_tfms(args.img_size, args.aug))
    val_ds = ChestXrayDataset(val_rows, val_split,
                              transform=make_eval_tfms(args.img_size))
    train_loader = build_loader(train_ds, per_gpu_bs, args.workers, True,
                                device.type == "cuda")
    val_loader = build_loader(val_ds, max(8, per_gpu_bs), args.workers, False,
                              device.type == "cuda")
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    print(f"训练 {len(train_rows)} 张 | 每卡 batch {per_gpu_bs} | 每轮 {steps_per_epoch} 步 "
          f"| 总步数 {total_steps} | warmup {warmup_steps} 步")
    print(f"验证 {len(val_rows)} 张（{len({r['Study_id'] for r in val_rows})} 个检查）")

    # ---------- 学习率（按 batch 缩放，这是 baseline1 最大的配方缺陷） ----------
    base_lr = (scaled_lr(LR, args.batch_size, args.lr_scale)
               if args.lr == "auto" else float(args.lr))
    print(f"骨干学习率 {base_lr:.2e}（LR_SCALE={args.lr_scale}, batch={args.batch_size}）"
          f" | 头 ×{args.head_lr_mult} | layer_decay={args.layer_decay}")

    def make_optimizer():
        groups = trainable_param_groups(
            model, base_lr=base_lr, head_lr_mult=args.head_lr_mult,
            weight_decay=args.weight_decay, layer_decay=args.layer_decay)
        return torch.optim.AdamW(groups, lr=base_lr)

    optimizer = make_optimizer()
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, decay=args.ema_decay, device=device)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    history = []
    best_auc = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        # 冻结/解冻（先解包，避免 DataParallel 没有 .backbone）
        if epoch == 0 and args.freeze_epochs > 0:
            set_backbone_trainable(model, False)
            print(f"[冻结] 前 {args.freeze_epochs} 轮只训分类头")
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            set_backbone_trainable(model, True)
            optimizer = make_optimizer()
            base_lrs = [g["lr"] for g in optimizer.param_groups]
            print("[解冻] 骨干恢复训练，优化器状态重建（前几轮只做线性探针，影响可忽略）")

        model.train()
        if args.freeze_epochs > 0 and epoch < args.freeze_epochs:
            get_inner(model).backbone.eval()     # 冻结时 BN/LN 也保持推理行为

        t0 = time.time()
        running, n_seen = 0.0, 0
        for imgs, _dicom, labels, view in train_loader:
            factor = lr_factor(global_step, total_steps, warmup_steps)
            for i, g in enumerate(optimizer.param_groups):
                g["lr"] = base_lrs[i] * factor

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            view = view.to(device, non_blocking=True) if args.use_view else None

            if args.mixup > 0:
                imgs, labels = mixup_batch(imgs, labels, alpha=args.mixup)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(imgs, view)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # ★ EMA 必须每 step 更新（写在这里，调用方没有写错的机会）
            ema.update(model)

            running += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            global_step += 1

        train_loss = running / max(1, n_seen)
        inner_ema = ema.ema
        auc, auc_gated, cls_aucs, _sp = evaluate(
            inner_ema, val_loader, val_rows, device, use_amp, args.gate_gamma)

        msg = (f"Epoch {epoch + 1}/{args.epochs} | loss {train_loss:.4f} | "
               f"study AUC {auc:.4f}")
        if auc_gated is not None:
            msg += f" | gated({args.gate_gamma}) {auc_gated:.4f}"
        msg += (f" | {time.time() - t0:.0f}s | EMA 更新 {ema.num_updates} 步 "
                f"(应={steps_per_epoch * (epoch + 1)})")
        print(msg)
        print("  每类 AUC:", format_auc_report(cls_aucs))

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "study_auc": auc, "study_auc_gated": auc_gated,
                        "per_class": [None if np.isnan(a) else round(float(a), 4)
                                      for a in cls_aucs],
                        "seconds": round(time.time() - t0, 1)})

        meta = {
            "model_state_dict": get_inner(model).state_dict(),
            "ema_state_dict": ema.state_dict(),
            "epoch": epoch + 1,
            "arch": args.arch,
            "impl": model.module.info["impl"] if n_gpu > 1 else model.info["impl"],
            "img_size": args.img_size,
            "src_size": SRC_SIZE,
            "num_classes": NUM_CLASSES,
            "view_dim": VIEW_DIM if args.use_view else 0,
            "pool": args.pool,
            "mean": list((0.485, 0.456, 0.406)),
            "std": list((0.229, 0.224, 0.225)),
            "study_auc": auc,
            "gate_gamma": args.gate_gamma,
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
        }
        torch.save(meta, os.path.join(out_dir, "last.pth"))
        if auc > best_auc:
            best_auc = auc
            torch.save(meta, os.path.join(out_dir, "best.pth"))
            print(f"  ** 新的最佳 study AUC {auc:.4f}，已保存 best.pth")

    # ---------- 收尾 ----------
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    expected = steps_per_epoch * args.epochs
    if ema.num_updates != expected:
        print(f"⚠ EMA 更新步数 {ema.num_updates} != 期望 {expected}，请检查训练循环")
    peak = max(history, key=lambda h: h["study_auc"])
    print("=" * 70)
    print(f"训练完成，权重在: {out_dir}")
    print(f"峰值 study AUC {peak['study_auc']:.4f}（第 {peak['epoch']} 轮）"
          f" | 末轮 {history[-1]['study_auc']:.4f}")
    if peak["epoch"] < args.epochs - 1:
        print(f"提示：峰值早于末轮 → 下一轮实验可把 --epochs 降到 "
              f"{min(args.epochs, peak['epoch'] + 2)}，省时间")
    print("=" * 70)


if __name__ == "__main__":
    main()
