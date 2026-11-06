# -*- coding: utf-8 -*-
"""批量推理：对 valid / test / 某个折的验证集跑模型，输出 npz（供 ensemble.py 集成）。

用法：
    # 测试集提交（默认读**原图** + draft 解码，与 cost.py 同源，规则第 7 条要求）
    python predict.py --split test --ckpt runs/xxx/ema_f12.pth --out runs/pred_test.npz

    # 官方 valid 上选型（默认读 data600 缓存，与训练评估口径一致）
    python predict.py --split valid --ckpt runs/xxx/best.pth --out runs/pred_valid.npz

    # 患者级 K 折的某折验证集（折的图在 data600/train 下）
    python predict.py --split train --csv runs/folds/fold_0_val.csv \
        --ckpt runs/f0/best.pth --out runs/f0/val_pred.npz

    # 多 ckpt 自动预测平均（Kaggle 方案的 checkpoint 平均）
    python predict.py --split test --ckpt a.pth b.pth --tta --out pred.npz

★ 关于 --src（图像来源，2026-11-02 修正）
------------------------------------------------
早先 predict.py 固定读 data600 小图，而 cost.py 读官方原图。两条重采样链不同，
实测 10 张里 5 张不一致、|Δp| 达 0.0122（容差是 1e-3）→ 核验必挂。
所以现在：
  --src auto    （默认）split=test 且没有 --csv 时读原图，其余读 data600 缓存
  --src original 强制读原图（最终提交、复现核验）
  --src cache    强制读 data600（只想快速看指标时用）
两条路都会走 `inference._load_one` 的 draft 解码，像素完全一致。

注意：
- 推理逻辑全部来自 inference.py，与 cost.py 共用同一份代码（规则第 7 条）；
- 同时写出 <out>_meta.json，记录本次推理的全部参数（含 src_size），便于 cost.py 对齐。
"""
import argparse
import json
import os

import numpy as np
import torch

from common import (aggregate_by_study, format_auc_report, macro_auc,
                    per_class_auc, prep_img_path, raw_img_path, read_csv_path,
                    read_split_csv)
from config import DATA_ROOT, USE_AMP
from inference import load_model_for_inference, predict_paths


def resolve_paths(rows, split, src):
    """按 --src 决定读原图还是读 data600 缓存。"""
    use_orig = src == "original"
    return [(raw_img_path(r, split, DATA_ROOT) if use_orig
             else prep_img_path(r, split)) for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["valid", "test", "train"],
                        help="test=提交；valid=官方验证；train=K 折的验证折（配 --csv）")
    parser.add_argument("--csv", default=None,
                        help="自定义行 CSV（K 折验证集用）。给了就不读官方 split 的 CSV")
    parser.add_argument("--ckpt", nargs="+", required=True,
                        help="一个或多个 ckpt（多个则预测平均）")
    parser.add_argument("--tta", action="store_true", help="启用 TTA（6 种确定性变换）")
    parser.add_argument("--src", default="auto", choices=["auto", "original", "cache"],
                        help="图像来源：auto/original/cache，见文件头说明")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=0,
                        help="覆盖 ckpt 里的推理尺寸（0=用 ckpt 原值）。"
                             "用途：448 训练好的权重直接在 384 推理，实测精度掉多少。"
                             "双模型 @448 在 T4 上约 111ms 有风险，@384 可省 26% 前向")
    parser.add_argument("--limit", type=int, default=0, help="冒烟测试：只推前 N 张")
    parser.add_argument("--out", required=True, help="输出 npz 路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    # ---------- 评估哪些行 ----------
    if args.csv:
        rows = read_csv_path(args.csv)
        print(f"行来自自定义 CSV: {args.csv}（{len(rows)} 张）")
    else:
        rows = read_split_csv(args.split)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[冒烟测试] 只推理 {len(rows)} 张")

    # ---------- 读哪一份图 ----------
    src = args.src
    if src == "auto":
        # 提交/核验场景必须读原图（与 cost.py 同源）；选型场景读缓存（与训练评估同口径）
        src = "original" if (args.split == "test" and not args.csv) else "cache"
    paths = resolve_paths(rows, args.split, src)
    print(f"图像来源: {'原图 + draft 解码' if src == 'original' else 'data600 缓存'}"
          f" | split={args.split}")

    # 抽样自检：防止路径配错（原图 1MB 左右，缓存 100KB 左右）
    if paths:
        size_mb = os.path.getsize(paths[0]) / 1024 / 1024
        print(f"首张图: {paths[0]}（{size_mb:.2f} MB）")

    # ---------- 推理 ----------
    probs_sum, meta = None, None
    for ckpt_path in args.ckpt:
        model, meta = load_model_for_inference(ckpt_path, device,
                                               img_size=args.img_size or None)
        print(f"已加载: {ckpt_path} | 骨干 {meta['backbone']} | "
              f"{meta['img_size']}px | draft {meta['src_size']}px"
              + ("（← --img-size 覆盖）" if args.img_size else ""))
        probs = predict_paths(model, paths, meta["img_size"], device,
                              tta=args.tta, batch_size=args.batch_size,
                              use_amp=use_amp, desc=f"[{args.split}]",
                              src_size=meta["src_size"])
        probs_sum = probs if probs_sum is None else probs_sum + probs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = probs_sum / len(args.ckpt)
    dicom_ids = [r["Dicom_id"] for r in rows]
    study_ids = [r["Study_id"] for r in rows]
    subject_ids = [r["Subject_id"] for r in rows]

    # 有标签的行（valid / 折的验证折）顺带打印 study 级 AUC（选型依据）
    if rows and "Predict_class" in rows[0] and args.split in ("valid", "train"):
        from common import encode_labels
        labels = encode_labels(rows)
        _, sp, sl = aggregate_by_study(rows, probs, labels=labels)
        print(f"[{args.split}] study 级 Macro AUC: {macro_auc(sl, sp):.4f}")
        print("  每类 AUC:", format_auc_report(per_class_auc(sl, sp)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, dicom_ids=np.array(dicom_ids),
             study_ids=np.array(study_ids), subject_ids=np.array(subject_ids),
             probs=probs.astype(np.float32))
    print(f"已保存 {args.out} | probs.shape={probs.shape}")

    # 记录推理参数：cost.py 与复现都要靠它保证一致
    meta_path = args.out.replace(".npz", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"ckpts": args.ckpt, "tta": bool(args.tta),
                   "img_size": meta["img_size"], "backbone": meta["backbone"],
                   "src": src, "src_size": meta["src_size"],
                   "split": args.split, "csv": args.csv,
                   "batch_size": args.batch_size, "num_images": len(rows)},
                  f, ensure_ascii=False, indent=2)
    print(f"推理参数已记录: {meta_path}")


if __name__ == "__main__":
    main()
