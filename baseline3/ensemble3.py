# -*- coding: utf-8 -*-
"""融合 + 提交：把若干 npz 合成 study 级概率，写出 submission.csv。

⚠ 一个必须说清的合规点（我此前建议错过的）
--------------------------------------
「秩平均融合」（rank average）在 Kaggle 很常见，但它需要**把整个测试集的预测放在一起排秩**，
即使用了跨多张测试图汇总出来的统计量。规则第 6 条明确禁止「跨多张测试图汇总统计量」，
而且 cost3.py 只跑 100 张，排出来的秩与 2218 张完全不同，**根本无法复现**。
所以本脚本只提供两种**逐样本**融合：
    mean  —— 概率平均
    logit —— 先转 log-odds 再平均（仍逐样本，等价于几何平均）
秩平均仅保留在 common3.rank_fuse 里供离线分析，不得用于提交。
"""
import argparse
import json
import os

import numpy as np

from common3 import (apply_normal_gating, check_submission, logit_fuse,
                     read_csv_path, read_split_csv, write_submission)
from config3 import AGG, NUM_CLASSES, PROB_DECIMALS


def load_npz_lut(path):
    z = np.load(path, allow_pickle=True)
    return {d: p for d, p in zip(z["dicom_ids"], z["probs"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", nargs="+", required=True, help="predict3.py 的输出（可多个）")
    p.add_argument("--split", default="test", choices=["test", "valid", "train"])
    p.add_argument("--csv", default=None, help="自定义行 CSV（用于折的验证集）")
    p.add_argument("--method", default="mean", choices=["mean", "logit"])
    p.add_argument("--agg", default=AGG, choices=["mean", "max"])
    p.add_argument("--gate", type=float, default=0.0, help="Normal Gating 的 gamma（0=关闭）")
    p.add_argument("--out", required=True)
    p.add_argument("--save-meta", default=None, help="记录融合参数（cost3 用它对齐）")
    args = p.parse_args()

    rows = read_csv_path(args.csv) if args.csv else read_split_csv(args.split)

    luts = [load_npz_lut(f) for f in args.npz]
    missing = [r["Dicom_id"] for r in rows if any(r["Dicom_id"] not in lut for lut in luts)]
    if missing:
        raise SystemExit(f"有 {len(missing)} 张图不在 npz 里（示例 {missing[:3]}），"
                         "检查 --npz 与 --split/--csv 是否匹配")
    print(f"读取 {len(luts)} 份预测 | 图像 {len(rows)} 张 | 融合方式 {args.method}")

    mats = [np.stack([lut[r["Dicom_id"]] for r in rows]) for lut in luts]
    if args.method == "logit":
        probs = logit_fuse(mats)
    else:
        probs = np.mean(mats, axis=0)
    print(f"图像级概率: shape={probs.shape} | 范围 [{probs.min():.4f}, {probs.max():.4f}] "
          f"| 均值 {probs.mean():.4f}")

    # 按 study 聚合（多图确定性平均 = 患者平滑，规则补充七允许）
    order, groups = [], {}
    for i, r in enumerate(rows):
        sid = r["Study_id"]
        if sid not in groups:
            groups[sid] = {"row": r, "idx": []}
            order.append(sid)
        groups[sid]["idx"].append(i)

    study_rows, study_probs = [], []
    for sid in order:
        idx = np.asarray(groups[sid]["idx"])
        p = probs[idx]
        study_rows.append(groups[sid]["row"])
        study_probs.append(p.mean(axis=0) if args.agg == "mean" else p.max(axis=0))
    study_probs = np.asarray(study_probs, dtype=np.float64)

    if args.gate > 0:
        before = study_probs.copy()
        study_probs = apply_normal_gating(study_probs, args.gate)
        delta = np.abs(study_probs - before).max()
        print(f"Normal Gating gamma={args.gate} 已应用（最大改动 {delta:.4f}；"
              "类 10 不变，仅压低其余 9 类）")

    n = write_submission(study_rows, study_probs, args.out,
                         all_classes=True, decimals=PROB_DECIMALS)
    print(f"已写入 {args.out}（{n} 行，全 {NUM_CLASSES} 类）")
    errs = check_submission(args.out, study_rows, all_classes=True)
    if errs:
        for e in errs:
            print("✗", e)
        raise SystemExit("提交文件自检未通过")
    print("提交文件自检通过（study 覆盖、行数、类别 id、概率范围均正确）")
    print(f"图像数 {len(rows)} -> study 数 {len(study_rows)}（聚合方式 {args.agg}）")

    if args.save_meta:
        with open(args.save_meta, "w", encoding="utf-8") as f:
            json.dump({"npz": args.npz, "method": args.method, "agg": args.agg,
                       "gate": args.gate, "submission": args.out,
                       "split": args.split, "csv": args.csv,
                       "num_images": len(rows), "num_study": len(study_rows)},
                      f, ensure_ascii=False, indent=2)
        print(f"融合参数已记录: {args.save_meta}（cost3.py --meta 用它对齐）")


if __name__ == "__main__":
    main()
