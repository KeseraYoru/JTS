# -*- coding: utf-8 -*-
"""实验汇总：一条命令看清所有 run 的成绩，以及集成候选组合。

为什么需要它
-----------
跑了一夜之后，`runs/` 下会有一堆目录。人工去翻 metrics.json 容易漏、容易记错峰值轮。
这个脚本做三件事：
1. 扫描所有 runs/*/metrics.json，按峰值排序，并标出**峰值轮**（判断是否该减 epochs）；
2. 打印关键类（类 1 纵隔心影增大，一直是最弱项）的 AUC；
3. 若给了 --npz，就在官方 valid 上算单模型 AUC 与**两两集成**（mean / logit），
   并按提升排序 —— 这样"换哪个成员进集成"不用猜。

注意：官方 valid 只有 859 个 study，**集成增益 <0.004 视为噪声**，不要凭 0.001 做决定。
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common3 import (aggregate_by_study, encode_labels, logit_fuse, macro_auc,
                     per_class_auc, read_split_csv)

# 参照线（都是实测值，不要改）
REFS = [
    ("convnext_base ImageNet-1k（全量, 12ep）", 0.7997),
    ("efficientnet_v2_m ImageNet-1k（全量）", 0.7993),
    ("双架构等权集成（平台 77.67）", 0.8027),
    ("fold0 探针：convnext_base @448 lr1e-4", 0.7612),
]


def scan_runs(runs_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        name = p.split(os.sep)[-2]
        try:
            hist = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not hist:
            continue
        peak = max(hist, key=lambda h: h["study_auc"])
        last = hist[-1]
        cls1 = peak.get("per_class", [None] * 10)[0]
        rows.append({
            "name": name,
            "peak": peak["study_auc"],
            "peak_ep": peak["epoch"],
            "last": last["study_auc"],
            "ep1": hist[0]["study_auc"],
            "cls1": cls1,
            "epochs": len(hist),
            "arch": next((h.get("arch") for h in hist if h.get("arch")), ""),
        })
    return rows


def print_runs(rows):
    if not rows:
        print("（没找到 runs/*/metrics.json）")
        return
    rows.sort(key=lambda r: -r["peak"])
    print(f"{'实验':<18}{'峰值':>8}{'峰值轮':>7}{'末轮':>8}{'类1':>8}  备注")
    print("-" * 62)
    for r in rows:
        note = ""
        if r["peak_ep"] <= r["epochs"] - 2:
            note = f"峰值早停 → 可减到 {r['peak_ep'] + 2} 轮"
        cls1 = f"{r['cls1']:.4f}" if isinstance(r["cls1"], float) else "  -"
        print(f"{r['name']:<18}{r['peak']:>8.4f}{r['peak_ep']:>7}{r['last']:>8.4f}"
              f"{cls1:>8}  {note}")
    print()
    print("参照：")
    for name, v in REFS:
        print(f"  {v:.4f}  {name}")


def ensemble_report(npz_paths, split="valid", topk=8):
    rows = read_split_csv(split)
    labels = encode_labels(rows)
    lut_rows = [r["Dicom_id"] for r in rows]

    mats, names = {}, []
    for p in npz_paths:
        try:
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue
        lut = {d: pr for d, pr in zip(z["dicom_ids"], z["probs"])}
        if not all(d in lut for d in lut_rows):
            print(f"跳过 {p}：与 {split} 的行不匹配")
            continue
        mats[p] = np.stack([lut[d] for d in lut_rows]).astype(np.float64)
        names.append(p)

    if not names:
        return

    def auc_of(probs):
        _, sp, sl = aggregate_by_study(rows, probs, how="mean", labels=labels)
        return macro_auc(sl, sp), per_class_auc(sl, sp)

    print(f"\n单模型（{split}，{len(rows)} 张 / "
          f"{len({r['Study_id'] for r in rows})} 个检查）")
    print("-" * 62)
    single = {}
    for p in names:
        a, _ = auc_of(mats[p])
        single[p] = a
        print(f"{os.path.basename(p):<34}{a:>8.4f}")

    if len(names) < 2:
        return
    print(f"\n两两集成（vs 最好的单模型）")
    print("-" * 62)
    combos = []
    best_single = max(single.values())
    for a, b in itertools.combinations(names, 2):
        for method, fn in (("mean", lambda x, y: (x + y) / 2),
                           ("logit", lambda x, y: logit_fuse([x, y]))):
            v, _ = auc_of(fn(mats[a], mats[b]))
            combos.append((v, method, a, b))
    combos.sort(reverse=True)
    for v, method, a, b in combos[:topk]:
        d = v - best_single
        flag = "✓" if d >= 0.004 else ("~" if d >= 0.001 else " ")
        print(f"{flag} {os.path.basename(a)[:16]:<17}+ {os.path.basename(b)[:16]:<17}"
              f"{method:<6}{v:.4f}  Δ{d:+.4f}")
    print("\n判据：Δ ≥ +0.004 才值得换（859 study，噪声 ±0.01，集成差值小得多但仍别信 0.001）")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=None, help="默认脚本上一级的 runs/")
    p.add_argument("--npz", nargs="*", default=None, help="valid 预测 npz，用于集成分析")
    p.add_argument("--split", default="valid")
    args = p.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = args.runs_dir or os.path.join(base, "runs")
    print("=" * 62)
    print(f"实验汇总: {runs_dir}")
    print("=" * 62)
    print_runs(scan_runs(runs_dir))

    npz = args.npz
    if npz is None:
        npz = sorted(glob.glob(os.path.join(runs_dir, "pv_*.npz")))
    if npz:
        ensemble_report(npz, split=args.split)


if __name__ == "__main__":
    main()
