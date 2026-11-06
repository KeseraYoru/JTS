# -*- coding: utf-8 -*-
"""知识蒸馏：把「5 折集成的判断」压进 1 个 convnext_base（合规，规则补充三允许）。

为什么需要它
------------
5 折集成 = 5 × 175MB(fp16) = 875MB，**超过 500MB 上限**。
但 5 折集成的好处是真实的（每张图被 4 个模型训练过 + 错误互补），不能白扔。
蒸馏就是在训练时用 5 个老师，推理时只跑 1 个学生 —— 体积回到 334MB(fp32)/175MB(fp16)。

合规依据（规则补充三）：
- 蒸馏本身允许（用官方训练数据训练学生模型）；
- 教师模型必须**非医学**预训练 → 我们的 5 个折模型都是 ImageNet + 官方数据训出来的，合规；
- 学生初始化权重同样非医学 → 用 SSL(官方数据自监督) + ImageNet，合规。

★ 软标签必须是 OOF（`tools/make_oof.py` 产出）
------------------------------------------------
教师若在自己训练过的图上出软标签，学生会学到"背答案"，在测试集上没有任何收益。
OOF 保证每一行训练图的软标签只来自没见过它的那个折模型。

损失设计
--------
    loss = alpha * 硬标签 Focal(加权, 抗不均衡)  +  (1-alpha) * 温度软标签 BCE
软标签部分用 T=2：先把概率转成 logit 再除以 T，算完乘 T²，这是标准做法，
能让软标签里的"类间相对关系"（比如"积液 0.8 但水肿 0.3"）更充分地被传递。

用法
----
    # ① 先给每折跑验证集预测（5 条命令，见 README 或手册）
    python predict.py --split train --csv runs/folds/fold_0_val.csv \
        --ckpt runs/f0/best.pth --out runs/f0/val_pred.npz

    # ② 拼 OOF（带三条断言）
    python tools/make_oof.py --preds runs/f{0,1,2,3,4}/val_pred.npz --out runs/oof_soft.npz

    # ③ 蒸馏
    python distill.py --soft runs/oof_soft.npz --epochs 18 --img-size 448 \
        --ssl runs/ssl/ssl_e200.pth --out runs/distill

判据（门 G4）：蒸馏模型在官方 valid 上 **≥ 单模型 − 0.002** 即算成功；
若明显掉点，说明软标签有问题（先查 OOF 覆盖率），或 alpha 太低（多信硬标签）。
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (aggregate_by_study, encode_labels, format_auc_report,
                    macro_auc, per_class_auc, prep_img_path, read_split_csv)
from config import (BATCH_SIZE, EMA_DECAY, FOCAL_ALPHA, FOCAL_GAMMA, FOCAL_MIX,
                    GRAD_CLIP, IMG_SIZE, NUM_CLASSES, NUM_WORKERS,
                    POS_WEIGHT_BETA, POS_WEIGHT_CAP, USE_AMP, WARMUP_RATIO)
from dataset import make_eval_tfms, make_train_tfms
from ema import ModelEMA
from losses import WeightedFocalBCELoss, calculate_pos_weights
from models import (BACKBONE_CHOICES, build_model, build_optimizer, get_model,
                    load_ssl_backbone)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, total_steps, warmup_ratio=WARMUP_RATIO, min_lr_ratio=0.01):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class DistillDataset(Dataset):
    """返回 (image, dicom_id, 硬标签, 软标签)。软标签来自 OOF。"""

    def __init__(self, rows, hard_labels, soft_table, transform, split="train"):
        self.rows = rows
        self.hard = hard_labels
        self.soft = soft_table
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(prep_img_path(r, self.split)) as im:
            img = self.transform(im.convert("RGB"))
        return (img, r["Dicom_id"], torch.from_numpy(self.hard[i]),
                torch.from_numpy(self.soft[r["Dicom_id"]]))


def soft_bce(logits, target_p, temp=2.0, eps=1e-6):
    """温度软标签 BCE：概率 → logit → /T，算完乘 T²（标准蒸馏缩放）。"""
    z_t = torch.logit(target_p.clamp(eps, 1.0 - eps))
    p_t = torch.sigmoid(z_t / temp)
    return F.binary_cross_entropy_with_logits(logits / temp, p_t) * (temp ** 2)


@torch.no_grad()
def validate(model, loader, device, use_amp, val_rows):
    model.eval()
    probs, labels = [], []
    for imgs, _d, lab, _s in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(lab.numpy())
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    _, sp, sl = aggregate_by_study(val_rows, probs, how="mean", labels=labels)
    return macro_auc(sl, sp), per_class_auc(sl, sp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soft", required=True, help="OOF 软标签 npz（tools/make_oof.py 产物）")
    p.add_argument("--backbone", default=None, choices=list(BACKBONE_CHOICES))
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--ssl", default=None, help="自监督初始化权重（与主线一致）")
    p.add_argument("--temp", type=float, default=2.0, help="软标签温度 T")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="硬标签损失权重（1-alpha 给软标签）；掉点就调高它")
    p.add_argument("--pos-cap", type=float, default=POS_WEIGHT_CAP)
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=os.path.join("runs", "distill"))
    args = p.parse_args()

    from config import BACKBONE, LR
    backbone = args.backbone or BACKBONE
    lr = args.lr or LR

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    print(f"设备 {device} | AMP {use_amp} | 骨干 {backbone} | "
          f"{args.img_size}px | T={args.temp} alpha={args.alpha}")

    # ---------- 数据 ----------
    rows = read_split_csv("train")
    if args.limit:
        rows = rows[:args.limit]
        print(f"[冒烟] 只用 {len(rows)} 张")
    hard = encode_labels(rows)

    z = np.load(args.soft, allow_pickle=True)
    soft_table = {d: pr for d, pr in zip(z["dicom_ids"], z["probs"].astype(np.float32))}
    n_missing = sum(1 for r in rows if r["Dicom_id"] not in soft_table)
    if n_missing:
        raise SystemExit(f"⚠ {n_missing}/{len(rows)} 行没有软标签 —— OOF 没覆盖全，先跑 tools/make_oof.py")
    print(f"软标签覆盖 {len(rows)}/{len(rows)} 行 ✓（教师=5折OOF，非医学，合规）")

    train_ds = DistillDataset(rows, hard, soft_table, make_train_tfms(args.img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"),
                              drop_last=len(train_ds) > args.batch_size,
                              persistent_workers=args.workers > 0)
    print(f"训练集 {len(train_ds)} 张 | 每轮 {len(train_loader)} 步")

    val_rows = read_split_csv("valid")
    val_loader = DataLoader(
        DistillDataset(val_rows, encode_labels(val_rows), soft_table,
                       make_eval_tfms(args.img_size), split="valid"),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), persistent_workers=args.workers > 0)
    print(f"验证集 {len(val_rows)} 张（{len({r['Study_id'] for r in val_rows})} 个检查）")

    # ---------- 模型 ----------
    model = build_model(backbone, pretrained=True)
    if args.ssl:
        model = load_ssl_backbone(model, args.ssl)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 块 GPU（DataParallel）")
        model = torch.nn.DataParallel(model)

    pos_w = calculate_pos_weights(hard, beta=POS_WEIGHT_BETA, cap=args.pos_cap).to(device)
    print("各类别 pos_weight:", [f"{w:.2f}" for w in pos_w.tolist()])
    criterion_hard = WeightedFocalBCELoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                                          pos_weights=pos_w, mix=FOCAL_MIX)
    optimizer = build_optimizer(get_model(model), lr=lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps)
    scaler = torch.amp.GradScaler(device="cuda" if use_amp else "cpu", enabled=use_amp)
    ema = ModelEMA(model, decay=EMA_DECAY, device=device)

    os.makedirs(args.out, exist_ok=True)
    print(f"输出目录: {args.out}")
    metrics, best = [], -1.0

    for epoch in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for imgs, _d, lab, sft in tqdm(train_loader, desc=f"[Distill] ep{epoch + 1}",
                                       leave=False, dynamic_ncols=True):
            imgs = imgs.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True).float()
            sft = sft.to(device, non_blocking=True).float()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(imgs)
                loss = (args.alpha * criterion_hard(logits, lab)
                        + (1 - args.alpha) * soft_bce(logits.float(), sft, temp=args.temp))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            ema.update(model)                      # ★ 必须每步，别放进 epoch 循环
            tot += loss.item() * imgs.size(0); n += imgs.size(0)

        auc, cls = validate(ema.ema, val_loader, device, use_amp, val_rows)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss {tot / max(1, n):.4f} | "
              f"valid study AUC {auc:.4f} | {time.time() - t0:.0f}s | "
              f"EMA 更新 {ema.num_updates} 步")
        print("  每类 AUC:", format_auc_report(cls))
        metrics.append({"epoch": epoch + 1, "loss": round(tot / max(1, n), 5),
                        "valid_study_auc": round(auc, 5)})
        if auc > best:
            best = auc
            torch.save({"model_state_dict": ema.ema.state_dict(),
                        "epoch": epoch + 1, "backbone": backbone,
                        "img_size": args.img_size,
                        "src_size": int(os.environ.get("CXR_SRC_SIZE", 600)),
                        "num_classes": NUM_CLASSES, "valid_study_auc": auc,
                        "distilled_from": "5-fold OOF ensemble"},
                       os.path.join(args.out, "best.pth"))
            print("  ** 新的最佳，已保存 best.pth **")
        with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n蒸馏完成，最佳 valid study AUC = {best:.4f}")
    print(f"权重: {os.path.join(args.out, 'best.pth')}")
    print("下一步：python predict.py --split test --ckpt "
          f"{os.path.join(args.out, 'best.pth')} --out runs/pred_distill.npz")


if __name__ == "__main__":
    main()
