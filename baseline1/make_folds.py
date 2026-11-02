# -*- coding: utf-8 -*-
"""按**患者**（Subject_id）切 K 折，生成 fold 用的 CSV。

为什么必须按患者切：同一个患者可能有多次检查（study），如果随机按检查切，
同一个患者的片子会同时出现在训练和验证里，模型等于提前看过答案 → 验证分数虚高。
官方 split 已经保证无 subject 交叉，我们自己切的时候必须保持这个性质。

用法：
    python make_folds.py --k 5 --seed 48 --out runs/folds

产出：
    runs/folds/fold_0_train.csv / fold_0_val.csv
    runs/folds/fold_1_train.csv / fold_1_val.csv
    ...

配合 train.py：
    python train.py --train-csv runs/folds/fold_0_train.csv \
                    --val-csv   runs/folds/fold_0_val.csv --out runs/cv_f0
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from common import encode_labels, read_split_csv
from config import CLASS_IDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--out", default=os.path.join("runs", "folds"))
    args = parser.parse_args()

    rows = read_split_csv("train")
    print(f"训练集：{len(rows)} 张图 / "
          f"{len({r['Study_id'] for r in rows})} 个检查 / "
          f"{len({r['Subject_id'] for r in rows})} 个患者")

    # 按患者分组
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["Subject_id"]].append(r)

    subjects = sorted(by_subject.keys())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(subjects)
    # 大的患者先分配，进「当前图片最少」的那一折（贪心均衡）
    subjects.sort(key=lambda s: -len(by_subject[s]))

    fold_rows = [[] for _ in range(args.k)]
    for s in subjects:
        target = min(range(args.k), key=lambda i: len(fold_rows[i]))
        fold_rows[target].extend(by_subject[s])

    os.makedirs(args.out, exist_ok=True)
    fieldnames = list(rows[0].keys())

    print("")
    print(f"{'fold':>5} {'训练图':>8} {'验证图':>8} {'验证检查':>9} {'验证患者':>9}")
    print("-" * 46)
    for i in range(args.k):
        val = fold_rows[i]
        train = [r for j in range(args.k) if j != i for r in fold_rows[j]]
        val_subj = {r["Subject_id"] for r in val}
        train_subj = {r["Subject_id"] for r in train}
        assert not (val_subj & train_subj), "患者跨折泄漏！"

        for name, subset in (("train", train), ("val", val)):
            path = os.path.join(args.out, f"fold_{i}_{name}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in subset:
                    w.writerow({k: r[k] for k in fieldnames})
        print(f"{i:>5} {len(train):>8} {len(val):>8} "
              f"{len({r['Study_id'] for r in val}):>9} {len(val_subj):>9}")

    # 类别覆盖检查：每折验证集里各类正样本数（太少的话该折 AUC 会很飘）
    print("")
    print("各折验证集的类别正样本数（应尽量均衡；极少说明该折估计不稳）：")
    print("fold " + " ".join(f"{c:>5}" for c in CLASS_IDS))
    for i in range(args.k):
        y = encode_labels(fold_rows[i])
        counts = y.sum(axis=0).astype(int)
        print(f"{i:>4} " + " ".join(f"{c:>5}" for c in counts))

    print("")
    print(f"✓ 已生成 {args.k} 折到 {args.out}/")
    print("  每折训练：python train.py --train-csv "
          f"{args.out}/fold_0_train.csv --val-csv {args.out}/fold_0_val.csv "
          "--out runs/cv_f0")


if __name__ == "__main__":
    main()
