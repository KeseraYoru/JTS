# -*- coding: utf-8 -*-
"""集成 + 生成提交文件：把多个模型的预测 npz 平均，按 study 聚合，写 submission.csv。

Kaggle 最终方案 = EVA-X(epoch5+6, TTA) 与 CheXFound(epoch5+6, 无TTA) 的预测平均。
本脚本对应推理 notebook 的最后一步，并补上了 Kaggle 不需要的 study 级聚合
（我们的评分单位是 Study_id，不是单张图）。

用法：
    python ensemble.py --split test \
        --inputs runs/evax_xxx/pred_test.npz runs/chexfound_xxx/pred_test.npz \
        --out submission.csv
    python ensemble.py --split valid --inputs ... --out val_check.csv   # 先自评
    python ensemble.py --split test --inputs a.npz b.npz --weights 2 1   # 加权平均
"""
import argparse
import os

import numpy as np

from common import (
    aggregate_by_study,
    check_submission,
    encode_labels,
    format_auc_report,
    macro_auc,
    per_class_auc,
    read_split_csv,
    write_submission,
)


def main():
    parser = argparse.ArgumentParser(description="多模型集成 + 生成 submission.csv")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--inputs", nargs="+", required=True, help="多个预测 npz")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="各 npz 的权重（默认等权平均）")
    parser.add_argument("--agg", default="mean", choices=["mean", "max"])
    parser.add_argument("--all-classes", action="store_true", default=True,
                        help="每个 study 输出全部 10 类概率（默认开启）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="不加 --all-classes 时，只输出概率 >= 阈值的类")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 张（需与 predict.py --limit 一致，"
                             "用于 Mac 冒烟测试；0 = 全部）")
    parser.add_argument("--out", required=True, help="输出 csv 路径")
    args = parser.parse_args()

    from config import PROB_DECIMALS, SUBMIT_ALL_CLASSES, SUBMIT_THRESHOLD

    rows = read_split_csv(args.split)
    if args.limit:
        rows = rows[:args.limit]
        print(f"[冒烟测试] 只处理 {len(rows)} 张")

    # ---------- 读取并对齐多个模型的预测 ----------
    weights = args.weights or [1.0] * len(args.inputs)
    if len(weights) != len(args.inputs):
        raise ValueError("--weights 数量必须和 --inputs 一致")
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    probs_sum, ref_ids = None, None
    for path, w in zip(args.inputs, weights):
        data = np.load(path, allow_pickle=True)
        # npz 里读出的是 numpy 字符串，统一转成普通 str 再比较
        ids = [str(x) for x in data["dicom_ids"]]
        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            raise ValueError(f"{path} 的图像顺序与第一个 npz 不一致")
        p = data["probs"] * w
        probs_sum = p if probs_sum is None else probs_sum + p
        print(f"读取 {path} (权重 {w:.3f})")

    if list(ref_ids) != [r["Dicom_id"] for r in rows]:
        raise ValueError("npz 里的图像顺序与 CSV 不一致，请检查是否用了同一个 split")

    # ---------- study 级聚合 ----------
    study_rows, study_probs, _ = aggregate_by_study(rows, probs_sum,
                                                    how=args.agg)
    print(f"图像数 {len(rows)} -> study 数 {len(study_rows)}（聚合方式 {args.agg}）")

    # ---------- valid 自评 ----------
    if args.split == "valid":
        labels = encode_labels(rows)
        _, _, study_labels = aggregate_by_study(rows, probs_sum, labels=labels,
                                                how=args.agg)
        print(f"[valid] study 级 Macro AUC: "
              f"{macro_auc(study_labels, study_probs):.4f}")
        print("  每类 AUC:", format_auc_report(per_class_auc(study_labels,
                                                            study_probs)))

    # ---------- 写提交文件 ----------
    all_classes = args.all_classes and SUBMIT_ALL_CLASSES
    threshold = SUBMIT_THRESHOLD if args.all_classes else args.threshold
    n = write_submission(study_rows, study_probs, args.out,
                         all_classes=all_classes, threshold=threshold,
                         decimals=PROB_DECIMALS)
    print(f"已写入 {args.out}（{n} 行，"
          f"{'全 10 类' if all_classes else f'阈值 {threshold}'}）")

    errors = check_submission(args.out, rows, all_classes=all_classes)
    if errors:
        print("提交文件自检未通过：")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)
    print("提交文件自检通过（study 覆盖、行数、类别 id、概率范围均正确）")


if __name__ == "__main__":
    main()
