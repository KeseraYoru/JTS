# -*- coding: utf-8 -*-
"""配对 bootstrap：判断配置 B 是否**真的**比 A 好（而不是噪声）。

为什么必须有这个工具
--------------------
官方 valid 只有 859 个 study，单个配置的 Macro AUC 标准误约 ±0.01。
两个配置差 0.005，完全可能是同一分布抽出来的两个样本 —— 直接比点估计就是在噪声里调参。
本工具对**同一批 study** 做有放回重采样 2000 次，算「配对差值」的 95% 区间：
区间跨 0 = 没有证据。

用法
----
    # K 折的某一折验证集（推荐：2211 个患者，比官方 valid 可靠 10 倍）
    python tools/bootstrap_ci.py \
        --a runs/A0/val_pred.npz --b runs/A5/val_pred.npz \
        --csv runs/folds/fold_0_val.csv

    # 官方 valid
    python tools/bootstrap_ci.py --a predA.npz --b predB.npz --split valid

    # 换聚合方式
    python tools/bootstrap_ci.py ... --agg max

约定
----
- 两个 npz 必须评估**同一批图、同一顺序**（predict.py 用同一个 csv 产出的就满足）；
- 输出 Δ 是 B − A，正数表示 B 更好；
- 只看最后那行结论：`有证据` 才允许进主线，`无证据（跨 0）` 一律当作"别动"。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (CLASS_IDS, aggregate_by_study, encode_labels,  # noqa: E402
                    macro_auc, per_class_auc, read_csv_path, read_split_csv)


def load_study_level(npz_path, rows, agg):
    """npz -> study 级 (probs, labels)。"""
    z = np.load(npz_path, allow_pickle=True)
    probs = z["probs"]
    if len(probs) != len(rows):
        raise SystemExit(
            f"{npz_path} 有 {len(probs)} 行，但 CSV 有 {len(rows)} 行 —— 不是同一批图。")
    _, sp, sl = aggregate_by_study(rows, probs, how=agg, labels=encode_labels(rows))
    return sp, sl, z["dicom_ids"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="基线 npz")
    p.add_argument("--b", required=True, help="对照 npz")
    p.add_argument("--csv", default=None, help="评估行对应 CSV（K 折验证用）")
    p.add_argument("--split", default="valid", choices=["valid", "train"])
    p.add_argument("--agg", default="mean", choices=["mean", "max"])
    p.add_argument("--n", type=int, default=2000, help="bootstrap 次数")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = read_csv_path(args.csv) if args.csv else read_split_csv(args.split)
    sp_a, sl, ids_a = load_study_level(args.a, rows, args.agg)
    sp_b, _, ids_b = load_study_level(args.b, rows, args.agg)
    if not (ids_a == ids_b).all():
        raise SystemExit("两个 npz 的图像顺序不一致，无法配对比较。")

    n = len(sl)
    auc_a, auc_b = macro_auc(sl, sp_a), macro_auc(sl, sp_b)
    print(f"评估 {n} 个 study（聚合方式 {args.agg}）")
    print(f"  A = {args.a}\n  B = {args.b}")
    print(f"  Macro AUC   A {auc_a:.4f}   B {auc_b:.4f}   Δ {auc_b - auc_a:+.4f}")

    # 逐类差值：定位是哪几个类在动（罕见类的小波动会藏在 macro 里）
    ca, cb = per_class_auc(sl, sp_a), per_class_auc(sl, sp_b)
    print("\n  每类 AUC（A → B）:")
    for cid, a, b in zip(CLASS_IDS, ca, cb):
        if np.isnan(a) or np.isnan(b):
            print(f"   {cid:>2}: 样本不足，跳过")
            continue
        mark = " *" if abs(b - a) >= 0.02 else ""
        print(f"   {cid:>2}: {a:.4f} -> {b:.4f}  ({b - a:+.4f}){mark}")

    # 配对 bootstrap：同一组 study 索引同时作用于 A 和 B，差值的方差才正确
    rng = np.random.default_rng(args.seed)
    diffs = np.empty(args.n)
    for i in range(args.n):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_auc(sl[idx], sp_b[idx]) - macro_auc(sl[idx], sp_a[idx])

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_win = float((diffs > 0).mean())
    print(f"\n  Δ 的 95% 区间: [{lo:+.4f}, {hi:+.4f}]   B 更优的概率 {p_win:.1%}")

    evidence = lo > 0 or hi < 0
    if evidence and lo > 0:
        verdict = "有证据：B 显著更好 → 可进主线"
    elif evidence and hi < 0:
        verdict = "有证据：B 显著更差 → 退化，别用"
    else:
        verdict = "无证据（区间跨 0）→ 按「没差别」处理，不要写'略有提升'"
    print(f"  【结论】{verdict}")
    print(f"\n  判据提醒：|Δ| < 0.008 且区间跨 0 时，只有'换更贵的方案'才需要谨慎；"
          f"\n  若 B 更便宜（如少训几轮），无证据即可采纳 B。")


if __name__ == "__main__":
    main()
