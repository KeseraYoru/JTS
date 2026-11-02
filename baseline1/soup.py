# -*- coding: utf-8 -*-
"""Model Soup：把多个模型（不同 seed / 不同 epoch）的**权重取平均**，合成一个模型。

为什么需要它：规则第 8 条要求「推理时实际加载的权重合计 ≤500MB」。一个
convnext_base 的 fp32 权重约 334MB，三个模型集成就是 1GB——**超限，交不了**。
但权重平均之后仍然**只有一个模型、一份体积**，却能吃到「多模型」的一部分收益
（多 seed 训练的方差被平均掉，排序更稳），是 500MB 约束下性价比最高的提分手段。

前提：所有 ckpt 必须是**同一个骨干、同一套参数名**（同一 ImageNet 初始化微调而来），
否则权重不在同一个「盆地」，平均会变差。脚本会先校验键名与形状，不一致直接报错。

用法：
    # 三个 seed 训完，各取最后一个 epoch 的 EMA 权重
    python soup.py --ckpt runs/s48/ema_f15.pth runs/s49/ema_f15.pth runs/s50/ema_f15.pth \
        --out runs/soup.pth --eval-split valid

    # 加权平均（权重会归一化）
    python soup.py --ckpt a.pth b.pth --weights 2 1 --out soup.pth
"""
import argparse
import os

import torch

from models import build_model  # noqa: F401  （保持与训练端相同的构建路径）


def load_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt, ckpt.get("model_state_dict", ckpt)


def merge(ckpts, weights=None):
    """对多个 state_dict 做加权平均；整数张量（如 num_batches_tracked）取第一个。"""
    states = []
    for c in ckpts:
        _full, sd = load_state(c)
        states.append(sd)

    ref_keys = list(states[0].keys())
    for path, sd in zip(ckpts[1:], states[1:]):
        if list(sd.keys()) != ref_keys:
            raise ValueError(f"{path} 的参数名与第一个 ckpt 不一致，无法平均")
        for k in ref_keys:
            if sd[k].shape != states[0][k].shape:
                raise ValueError(f"{path} 的 {k} 形状不一致: "
                                 f"{tuple(sd[k].shape)} vs {tuple(states[0][k].shape)}")

    if weights is None:
        weights = [1.0] * len(states)
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()

    merged = {}
    for k in ref_keys:
        t0 = states[0][k]
        if not torch.is_floating_point(t0):
            merged[k] = t0.clone()
            continue
        acc = torch.zeros_like(t0, dtype=torch.float64)
        for wi, sd in zip(w, states):
            acc += sd[k].to(torch.float64) * float(wi)
        merged[k] = acc.to(t0.dtype)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval-split", default=None, choices=["valid"],
                        help="可选：平均后在 valid 上评估 study 级 AUC")
    args = parser.parse_args()

    if args.weights and len(args.weights) != len(args.ckpt):
        raise SystemExit("--weights 数量必须和 --ckpt 一致")

    print(f"合并 {len(args.ckpt)} 个模型：")
    for c in args.ckpt:
        full, _ = load_state(c)
        print(f"  - {c}（epoch {full.get('epoch')}, valid {full.get('study_auc')}）")

    merged = merge(args.ckpt, args.weights)
    head, _ = load_state(args.ckpt[0])
    meta = {k: head.get(k) for k in ("backbone", "img_size", "num_classes") if head.get(k)}
    meta["soup_from"] = args.ckpt
    meta["soup_weights"] = args.weights

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model_state_dict": merged, **meta}, args.out)
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"✓ 已保存 {args.out}（{size_mb:.1f} MB，仍是单模型体积）")

    if args.eval_split:
        from common import (aggregate_by_study, encode_labels, format_auc_report,
                            macro_auc, per_class_auc, prep_img_path,
                            read_split_csv)
        from inference import load_model_for_inference, predict_paths

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = read_split_csv(args.eval_split)
        paths = [prep_img_path(r, args.eval_split) for r in rows]
        model, m = load_model_for_inference(args.out, device)
        probs = predict_paths(model, paths, m["img_size"], device, tta=True,
                              batch_size=16, use_amp=(device.type == "cuda"),
                              desc="[soup]")
        labels = encode_labels(rows)
        _, sp, sl = aggregate_by_study(rows, probs, labels=labels)
        print(f"[{args.eval_split}] soup study 级 Macro AUC: {macro_auc(sl, sp):.4f}")
        print("  每类 AUC:", format_auc_report(per_class_auc(sl, sp)))
        print("对比：上面各 ckpt 的 valid 分数在 merged 前的列表里（study_auc）")


if __name__ == "__main__":
    main()
