# -*- coding: utf-8 -*-
"""批量推理：把图像级概率写成 npz（供 ensemble3 融合/提交）。

用法
----
    # 测试集（默认读原图 + draft 解码，与 cost3 同源）
    python predict3.py --split test --ckpt runs/full_dino/best.pth --out runs/pred_test_dino.npz

    # 多个 ckpt 直接平均（同架构不同 seed / 同架构不同轮次）
    python predict3.py --split test --ckpt a.pth b.pth --out runs/pred_ab.npz

    # 验证集选型（默认读预处理缓存，与训练评估同口径）
    python predict3.py --split valid --ckpt runs/x/best.pth --out runs/pv.npz

为什么要区分 --src
----------------
训练与验证读预处理图（data600/1024），提交与核验读原图 —— 两条链必须由同一份
`infer3.load_image` 处理，否则重采样次数不同，概率差 0.01 量级。
"""
import argparse
import json
import os

import numpy as np
import torch

from common3 import (aggregate_by_study, encode_labels, format_auc_report,
                     macro_auc, per_class_auc, prep_img_path, raw_img_path,
                     read_csv_path, read_split_csv)
from config3 import DATA_ROOT, USE_AMP
from infer3 import load_from_ckpt, predict_paths

TTA_KINDS = {"none": ("base",), "hflip": ("base", "hflip"),
             "both": ("base", "hflip", "small")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["valid", "test", "train"])
    p.add_argument("--csv", default=None, help="自定义行 CSV（折的验证集用）")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--tta", default="none", choices=list(TTA_KINDS))
    p.add_argument("--src", default="auto", choices=["auto", "original", "cache"])
    p.add_argument("--img-size", type=int, default=0, help="覆盖 ckpt 里的推理尺寸（0=用 ckpt 值）")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--disable-ema", action="store_true", help="用原始权重而非 EMA")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    rows = read_csv_path(args.csv) if args.csv else read_split_csv(args.split)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[冒烟] 只推理 {len(rows)} 张")

    src = args.src
    if src == "auto":
        src = "original" if (args.split == "test" and not args.csv) else "cache"
    use_raw = src == "original"
    paths = [(raw_img_path(r, args.split, DATA_ROOT) if use_raw
              else prep_img_path(r, args.split)) for r in rows]
    print(f"图像来源: {'原图 + draft 解码' if use_raw else '预处理缓存'} | split={args.split}")
    if paths:
        if not os.path.exists(paths[0]):
            raise SystemExit(
                f"图片不存在: {paths[0]}\n"
                "提交/核验必须读原图（与 cost3 同源），所以 test 默认走原图。\n"
                "若本机只有预处理缓存、没有原始大图，可先用 --src cache 做流程冒烟，"
                "但**正式提交前必须用原图重跑**，否则 cost3 一致性核验会挂。")
        mb = os.path.getsize(paths[0]) / 1024 / 1024
        print(f"首张图: {paths[0]}（{mb:.2f} MB）")

    probs_sum, metas = None, []
    for ckpt in args.ckpt:
        model, meta = load_from_ckpt(ckpt, device, use_ema=not args.disable_ema,
                                     img_size_override=args.img_size or None)
        metas.append(meta)
        if meta["loaded_keys"] < 0.95 * meta["total_keys"]:
            print(f"⚠ {ckpt} 只匹配 {meta['loaded_keys']}/{meta['total_keys']} 键，"
                  "权重可能与结构不符，请先跑 train3.py --selftest 核对")
        print(f"已加载: {ckpt} | {meta['impl']}/{meta['arch']} | {meta['img_size']}px | "
              f"draft {meta['src_size']}px | view_dim {meta['view_dim']} | pool {meta['pool']}")

        probs = predict_paths(model, paths, rows, meta["img_size"], device,
                              tta_kinds=TTA_KINDS[args.tta], batch_size=args.batch_size,
                              use_amp=use_amp, desc=f"[{args.split}]",
                              src_size=meta["src_size"])
        probs_sum = probs if probs_sum is None else probs_sum + probs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = (probs_sum / len(args.ckpt)).astype(np.float32)

    # 有标签时顺带报 study 级指标（选型依据）
    if rows and "Predict_class" in rows[0] and args.split in ("valid", "train"):
        labels = encode_labels(rows)
        _, sp, sl = aggregate_by_study(rows, probs, labels=labels)
        print(f"[{args.split}] study 级 Macro AUC: {macro_auc(sl, sp):.4f}")
        print("  每类 AUC:", format_auc_report(per_class_auc(sl, sp)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             dicom_ids=np.array([r["Dicom_id"] for r in rows]),
             study_ids=np.array([r["Study_id"] for r in rows]),
             subject_ids=np.array([r["Subject_id"] for r in rows]),
             probs=probs)
    print(f"已保存 {args.out} | probs.shape={probs.shape}")

    meta_path = args.out.replace(".npz", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "ckpts": args.ckpt,
            "tta": args.tta,
            "src": src,
            "split": args.split,
            "csv": args.csv,
            "img_size": metas[0]["img_size"],
            "src_size": metas[0]["src_size"],
            "archs": [m["arch"] for m in metas],
            "fusion": "prob_mean",
            "batch_size": args.batch_size,
            "num_images": len(rows),
        }, f, ensure_ascii=False, indent=2)
    print(f"推理参数已记录: {meta_path}")


if __name__ == "__main__":
    main()
