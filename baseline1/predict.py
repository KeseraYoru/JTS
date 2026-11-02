# -*- coding: utf-8 -*-
"""批量推理：对 valid / test 全量跑模型，输出 npz（供 ensemble.py 集成）。

用法：
    python predict.py --split test  --ckpt runs/xxx/ema_f11.pth runs/xxx/ema_f12.pth --tta --out runs/xxx/pred_test.npz
    python predict.py --split valid --ckpt runs/xxx/ema_f12.pth --tta --out runs/xxx/pred_valid.npz

注意：
- 多个 --ckpt 会自动「预测取平均」，这正是 Kaggle 方案的 checkpoint 平均；
- 推理逻辑全部来自 inference.py，与 cost.py 共用同一份代码（规则第 7 条）；
- 同时写出 <out>_meta.json，记录本次推理的全部参数，便于 cost.py 与复现对齐。
"""
import argparse
import json
import os

import numpy as np
import torch

from common import prep_img_path, read_split_csv
from inference import load_model_for_inference, predict_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--ckpt", nargs="+", required=True,
                        help="一个或多个 ckpt（多个则预测平均）")
    parser.add_argument("--tta", action="store_true", help="启用 TTA（6 种确定性变换）")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="冒烟测试：只推前 N 张")
    parser.add_argument("--out", required=True, help="输出 npz 路径")
    args = parser.parse_args()

    from config import USE_AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    rows = read_split_csv(args.split)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[冒烟测试] 只推理 {len(rows)} 张")
    paths = [prep_img_path(r, args.split) for r in rows]

    probs_sum, meta = None, None
    for ckpt_path in args.ckpt:
        model, meta = load_model_for_inference(ckpt_path, device)
        print(f"已加载: {ckpt_path} | 骨干 {meta['backbone']} | {meta['img_size']}px")
        probs = predict_paths(model, paths, meta["img_size"], device,
                              tta=args.tta, batch_size=args.batch_size,
                              use_amp=use_amp, desc=f"[{args.split}]")
        probs_sum = probs if probs_sum is None else probs_sum + probs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = probs_sum / len(args.ckpt)
    dicom_ids = [r["Dicom_id"] for r in rows]
    study_ids = [r["Study_id"] for r in rows]
    subject_ids = [r["Subject_id"] for r in rows]

    # valid 上顺带打印 study 级 AUC（选型依据）
    if args.split == "valid":
        from common import aggregate_by_study, encode_labels, format_auc_report, \
            macro_auc, per_class_auc
        labels = encode_labels(rows)
        _, sp, sl = aggregate_by_study(rows, probs, labels=labels)
        print(f"[valid] study 级 Macro AUC: {macro_auc(sl, sp):.4f}")
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
                   "split": args.split, "batch_size": args.batch_size,
                   "num_images": len(rows)}, f, ensure_ascii=False, indent=2)
    print(f"推理参数已记录: {meta_path}")


if __name__ == "__main__":
    main()
