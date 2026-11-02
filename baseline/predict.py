# -*- coding: utf-8 -*-
"""推理脚本：对 valid 或 test 跑 TTA 推理，输出图像级概率 npz。

Kaggle 方案的推理配置：
  EVA-X:    第 5、6 两个 epoch 的权重 + TTA（6 种变换取平均）
  CheXFound: 第 5、6 两个 epoch 的权重，不用 TTA

用法：
    python predict.py --model evax --split test --tta \
        --ckpt runs/evax_xxx/ema_f5.pth runs/evax_xxx/ema_f6.pth \
        --out runs/evax_xxx/pred_test.npz
    python predict.py --model chexfound --split test \
        --ckpt runs/chexfound_xxx/ema_f5.pth runs/chexfound_xxx/ema_f6.pth \
        --out runs/chexfound_xxx/pred_test.npz
    python predict.py --model evax --split valid --tta --ckpt ...   # 顺带打印验证分数

多个 --ckpt 会像 Kaggle 一样对预测概率取平均（不是平均权重）。
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    aggregate_by_study,
    format_auc_report,
    macro_auc,
    per_class_auc,
    read_split_csv,
)
from dataset import ChestXrayDataset, make_tta_tfms, make_val_tfms
from models import build_model, default_batch_size, model_img_size


def load_weights(model, ckpt_path, device):
    """把 ckpt 加载进模型。支持三种保存格式：
    1. {"ema_state_dict": ...}      我们的 EMA 权重
    2. {"model_state_dict": ...}    我们的普通权重
    3. 裸 state_dict                Kaggle 原始格式
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("ema_state_dict", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state_dict = ckpt[key]
            break
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    print(f"已加载权重: {ckpt_path}")


@torch.no_grad()
def predict_with_tta(model, dataset, tta_tfms, batch_size, device, use_amp):
    """Kaggle 原版 TTA 推理：对每种变换各推一遍，最后对概率取平均。"""
    model.eval()
    all_ids, fold_probs = None, []
    for tfm in tta_tfms:
        dataset.transform = tfm
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=(device.type == "cuda"))
        out, ids = [], []
        for images, img_id, _labels in tqdm(loader, desc="[TTA]"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
            out.append(torch.sigmoid(logits).float().cpu())
            ids.extend(img_id)
        fold_probs.append(torch.cat(out, dim=0).numpy())
        if all_ids is None:
            all_ids = ids
    return all_ids, np.mean(fold_probs, axis=0)


@torch.no_grad()
def predict_plain(model, dataset, batch_size, device, use_amp):
    """无 TTA 的标准推理（CheXFound 用）。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))
    out, ids = [], []
    for images, img_id, _labels in tqdm(loader, desc="[Infer]"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
        out.append(torch.sigmoid(logits).float().cpu())
        ids.extend(img_id)
    return ids, torch.cat(out, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="TTA 推理并输出 npz")
    parser.add_argument("--model", default="evax", choices=["evax", "chexfound"])
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--ckpt", nargs="+", required=True,
                        help="一个或多个权重文件（概率取平均）")
    parser.add_argument("--tta", action="store_true", help="启用 TTA")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--agg", default="mean", choices=["mean", "max"])
    parser.add_argument("--limit", type=int, default=0,
                        help="只推理前 N 张（0 = 全部；Mac 无 GPU 时小样本验证用）")
    parser.add_argument("--out", required=True, help="输出 npz 路径")
    args = parser.parse_args()

    from config import USE_AMP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    batch_size = args.batch_size or default_batch_size(args.model) * 2
    img_size = model_img_size(args.model)

    rows = read_split_csv(args.split)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[冒烟测试] 只推理 {len(rows)} 张")
    with_labels = args.split == "valid"
    dataset = ChestXrayDataset(rows, args.split, transform=None,
                               with_labels=with_labels)
    tta_tfms = make_tta_tfms(img_size) if args.tta else None
    if tta_tfms is None:
        # 无 TTA 路径：必须显式设置验证预处理（resize + 归一化），
        # 否则 dataset.transform 为 None，会把未归一化的原始张量喂给模型
        dataset.transform = make_val_tfms(img_size)

    # 多个 ckpt：各自推理后对概率取平均（Kaggle 的 epoch5+epoch6 策略）
    probs_sum = None
    dicom_ids = None
    for ckpt_path in args.ckpt:
        model = build_model(args.model, pretrained=False).to(device)
        load_weights(model, ckpt_path, device)
        if tta_tfms is not None:
            ids, probs = predict_with_tta(model, dataset, tta_tfms,
                                          batch_size, device, use_amp)
        else:
            ids, probs = predict_plain(model, dataset, batch_size, device,
                                       use_amp)
        if dicom_ids is None:
            dicom_ids = ids
        else:
            assert ids == dicom_ids, "不同 ckpt 的图像顺序不一致！"
        probs_sum = probs if probs_sum is None else probs_sum + probs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = probs_sum / len(args.ckpt)
    study_ids = [r["Study_id"] for r in rows]
    subj_ids = [r["Subject_id"] for r in rows]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             dicom_ids=np.array(dicom_ids),
             study_ids=np.array(study_ids),
             subject_ids=np.array(subj_ids),
             probs=probs)
    print(f"已保存: {args.out}  probs.shape={probs.shape}")

    # valid  split 顺手打印 study 级分数，用来给模型选型
    if with_labels:
        from common import encode_labels
        labels = encode_labels(rows)
        _, study_probs, study_labels = aggregate_by_study(
            rows, probs, labels=labels, how=args.agg
        )
        print(f"[{args.split}] study 级 Macro AUC: "
              f"{macro_auc(study_labels, study_probs):.4f}")
        print("  每类 AUC:", format_auc_report(per_class_auc(study_labels,
                                                            study_probs)))


if __name__ == "__main__":
    main()
