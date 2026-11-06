# -*- coding: utf-8 -*-
"""把 5 折的验证集预测拼成完整的 OOF 软标签（蒸馏的教师答案）。

为什么必须这么做（做错整个蒸馏就是假的）
--------------------------------------
想蒸馏 5 折集成，就得先有"教师对训练图的预测"。但每个折模型都**见过自己那 80% 的训练图**，
直接拿它预测训练图 → 教师在自己见过的图上过度自信 → 学生学到的是"背答案"，
在测试集上毫无收益。

正确规则：**每一行训练图的软标签，只由没见过它的那个折模型提供** —— 这正是 OOF：
- 行 r 在 fold_i 的 val 里 → 由 f_i 预测（f_i 训练时没见过 r）
- 5 折的 val 恰好把全部 30344 行划分完，每行**恰好被覆盖一次**

所以本脚本的核心不是拼接，而是三条断言：
1. 每行恰好出现一次（重复 = 折划分有重叠，会泄漏）；
2. 全部行都被覆盖（缺失 = 该行没有软标签）；
3. 没有多余的行（多余 = 折文件与官方 train 对不上）。

用法：
    python tools/make_oof.py --preds runs/f0/val_pred.npz runs/f1/val_pred.npz \
        runs/f2/val_pred.npz runs/f3/val_pred.npz runs/f4/val_pred.npz \
        --out runs/oof_soft.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import read_split_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", nargs="+", required=True,
                   help="每折在各自 val 上的预测 npz（predict.py --split train --csv fold_i_val.csv 的产物）")
    p.add_argument("--out", default="runs/oof_soft.npz")
    args = p.parse_args()

    if len(args.preds) < 2:
        raise SystemExit("至少要给 2 折的预测")

    table, dup = {}, []
    for path in args.preds:
        if not os.path.exists(path):
            raise SystemExit(f"找不到 {path} —— 该折的预测还没跑")
        z = np.load(path, allow_pickle=True)
        ids, probs = z["dicom_ids"], z["probs"]
        if probs.shape[1] != 10:
            raise SystemExit(f"{path} 的 probs 维度是 {probs.shape[1]}，应为 10")
        for d, pr in zip(ids, probs):
            if d in table:
                dup.append(d)
            else:
                table[d] = pr.astype(np.float32)
            if d in table and d in dup:
                continue

    # 断言 1：不能有重复（重复意味着同一张图被两个折模型都当"没见过"，说明折划分有泄漏）
    if dup:
        raise SystemExit(
            f"⚠ {len(dup)} 张图被多个折模型预测过（如 {dup[:3]}）。\n"
            f"  说明折划分有重叠或张冠李戴，OOF 不成立，蒸馏会泄漏。请先检查 runs/folds/。")

    rows = read_split_csv("train")
    dicoms = [r["Dicom_id"] for r in rows]
    dicom_set = set(dicoms)

    missing = [d for d in dicoms if d not in table]
    extra = [d for d in table if d not in dicom_set]

    print(f"官方 train: {len(dicoms)} 张")
    print(f"OOF 覆盖  : {len(table)} 张")
    print(f"缺失      : {len(missing)} 张" + (f"（如 {missing[:3]}）" if missing else ""))
    print(f"多余      : {len(extra)} 张" + (f"（如 {extra[:3]}）" if extra else ""))

    if missing:
        raise SystemExit("⚠ 有图没有被任何折的 val 覆盖 → 这些学生样本没有软标签，不能蒸馏。")
    if extra:
        raise SystemExit("⚠ 有 OOF 行不在官方 train 里 → 折文件与 train.csv 对不上。")

    probs = np.stack([table[d] for d in dicoms]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, dicom_ids=np.array(dicoms), probs=probs)

    print(f"\n✓ 已写 {args.out} | probs.shape={probs.shape}")
    print(f"  概率范围 [{probs.min():.4f}, {probs.max():.4f}] | 均值 {probs.mean():.4f}")
    print("  下一步: python distill.py --soft runs/oof_soft.npz ...")


if __name__ == "__main__":
    main()
