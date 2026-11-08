# -*- coding: utf-8 -*-
"""模型两两组合搜索：找出"哪一对最好"，并且**不让挑选本身骗人**。

为什么不能直接"取 valid 最高的那对"
----------------------------------
官方 valid 只有 859 个 study，单个组合的 Macro AUC 标准误约 ±0.01。
在 15 对里挑最大值，等于从 15 个噪声样本里挑最大的那个 —— 期望虚高约 +0.005。
我们已经被这件事咬过一次：large384 的 valid 是 0.8059（全场最高），
平台只有 77.37，反而不如 valid 0.8027 的双模型（77.67）。

所以本工具做三件事：
1. 把**所有单模型、所有两两组合**都算出来并排序（让你看到全景）；
2. 对每一对算「相对现役冠军的配对 bootstrap 差值区间」（判断差异是否真实）；
3. 按**预先写死的规则**给推荐，而不是取最大值：
   - 规则 A（一标准误原则）：在"与第一名差距 < 一个标准误(~0.008)"的组合里，
     选**成员最少 / 体积最小**的那个；
   - 规则 B（必须超越现役）：只有当某对相对现役的 Δ 下界 > 0 才建议换。

用法
----
    # 自动扫描 runs/*/pv*.npz（推荐）
    python tools/pair_search.py --auto \
        --champion convnext_base effv2m

    # 手工指定成员
    python tools/pair_search.py \
        --member convnext_base=runs/full_base/pv2.npz \
        --member effv2m=runs/full_effv2m/pv.npz \
        --member large448=runs/large448_full/pv.npz \
        --champion convnext_base effv2m

"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (aggregate_by_study, encode_labels,  # noqa: E402
                    macro_auc, read_split_csv)
from config import NUM_CLASSES  # noqa: E402

# 已知骨干的 fp16 体积（MB），用于体积核算；认不出来就记 0
SIZE_HINT = {
    # 前端是实验目录名（runs/*），后端是骨干名；按插入顺序匹配，先具体后笼统
    "large448": 377, "large384": 377, "convnext_large": 377,
    "full_effv2m": 108, "effv2m": 108, "efficientnet_v2_m": 108,
    "efficientnet_v2_s": 41, "convnext_small": 94,
    "res50": 45, "resnet50": 45, "swin_t": 53, "swin_b": 167,
    "full_base": 169, "full_1024": 169, "convnext_base": 169,
    "dino": 164, "large": 377,
    # baseline3 的成员目录名（2026-11-08 新增，否则体积会显示成 0MB）
    "in22k30": 167, "in22k_384": 167,
    "effv2s30": 41, "effv2s_384": 41,
    "cnxsmall30": 94, "cnx_small_384": 94,
    "eff30": 108, "effv2m_384": 108,
    "swinb30": 167, "swin_b45": 167,
}


def guess_size(name):
    low = name.lower()
    for k, v in SIZE_HINT.items():
        if k in low:
            return v
    return 0


def load_study_level(path, rows, labels):
    """npz -> study 级概率 (n_study, 10)。与评测口径完全一致（mean 聚合）。"""
    z = np.load(path, allow_pickle=True)
    probs = z["probs"].astype(np.float64)
    if probs.shape[0] != len(rows):
        raise SystemExit(f"{path} 行数 {probs.shape[0]} 与 CSV {len(rows)} 不一致")
    _, sp, sl = aggregate_by_study(rows, probs, how="mean", labels=labels)
    return sp, sl


def macro(sp, sl):
    return macro_auc(sl, sp)


def bootstrap_delta(sp_a, sp_b, sl, n=2000, seed=0):
    """配对 bootstrap：同一批 study 重采样，算 Δ = B − A 的分布。"""
    rng = np.random.default_rng(seed)
    n_s = len(sl)
    deltas = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_s, n_s)
        ya, yb = sl[idx], sl[idx]
        deltas[i] = macro(sp_b[idx], yb) - macro(sp_a[idx], ya)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi), float((deltas > 0).mean())


def main():
    p = argparse.ArgumentParser(description="两两组合搜索（带选择偏差防护）")
    p.add_argument("--member", action="append", default=[],
                   help="名字=路径，可重复。与 --auto 二选一")
    p.add_argument("--auto", action="store_true",
                   help="自动扫描 runs/*/pv*.npz（取每个目录里最新的一份）")
    p.add_argument("--champion", nargs="+", default=None,
                   help="现役冠军的成员名（用于算 Δ）；默认 convnext_base effv2m")
    p.add_argument("--split", default="valid", choices=["valid"])
    p.add_argument("--n", type=int, default=2000, help="bootstrap 次数")
    p.add_argument("--out", default=None, help="把结果另存为 json")
    args = p.parse_args()

    rows = read_split_csv(args.split)
    labels = encode_labels(rows)

    # ---------- 收集成员 ----------
    members = {}
    if args.auto:
        for d in sorted(glob.glob("runs/*")):
            cands = sorted(glob.glob(os.path.join(d, "pv*.npz")),
                           key=os.path.getmtime)
            if cands:
                members[os.path.basename(d)] = cands[-1]
    for spec in args.member:
        if "=" not in spec:
            raise SystemExit(f"--member 需要 名字=路径 形式，收到 {spec}")
        name, path = spec.split("=", 1)
        members[name] = path

    if len(members) < 2:
        raise SystemExit("可用成员少于 2 个，先跑出 valid 预测（predict.py --split valid）")

    # ---------- 单模型 ----------
    sp, sl = {}, None
    for name, path in members.items():
        s, y = load_study_level(path, rows, labels)
        sp[name], sl = s, y
    names = list(sp)

    print("=" * 78)
    print(f"单模型（{len(sl)} 个 study，mean 聚合）")
    print(f"{'成员':<22}{'valid':>8}{'fp16体积':>10}")
    print("-" * 78)
    singles = []
    for n in names:
        a = macro(sp[n], sl)
        singles.append((n, a))
        print(f"{n:<22}{a:>8.4f}{guess_size(n):>8}MB")
    singles.sort(key=lambda x: -x[1])

    # ---------- 相关性（互补性） ----------
    print("\n成员相关性（越低越互补；>0.98 基本是同一个模型）")
    print("-" * 78)
    print("        " + "".join(f"{n[:8]:>10}" for n in names))
    for i, a in enumerate(names):
        line = f"{a[:8]:<8}"
        for b in names:
            c = np.corrcoef(sp[a].ravel(), sp[b].ravel())[0, 1]
            line += f"{c:>10.3f}"
        print(line)

    # ---------- 两两组合 ----------
    pairs = []
    for a, b in itertools.combinations(names, 2):
        m = (sp[a] + sp[b]) / 2
        pairs.append((f"{a} + {b}", macro(m, sl), [a, b]))
    pairs.sort(key=lambda x: -x[1])

    print(f"\n两两组合（等权平均，共 {len(pairs)} 对，按 valid 排序）")
    print(f"{'组合':<44}{'valid':>8}{'体积':>8}")
    print("-" * 78)
    for label, a, ms in pairs:
        size = sum(guess_size(x) for x in ms)
        print(f"{label:<44}{a:>8.4f}{size:>6}MB")

    # ---------- 冠军与 Δ ----------
    champ = args.champion or ["convnext_base", "effv2m"]
    champ = [c for c in champ if c in sp]
    if len(champ) < 1:
        print("\n⚠ 冠军成员名不在可用列表里，跳过 Δ 计算")
        return

    sp_champ = np.mean([sp[c] for c in champ], axis=0) if len(champ) > 1 else sp[champ[0]]
    a_champ = macro(sp_champ, sl)
    print(f"\n现役冠军：{' + '.join(champ)}  valid {a_champ:.4f}")

    print(f"\n相对冠军的配对 bootstrap（Δ>0 表示比冠军好；区间跨 0 = 没证据）")
    print(f"{'组合':<40}{'valid':>8}{'Δ':>8}{'95%区间':>20}{'P(更好)':>9}")
    print("-" * 90)
    detail = []
    for label, a, ms in pairs:
        if set(ms) == set(champ):
            print(f"{label:<40}{a:>8.4f}{0.0:>+8.4f}{'        （基准）':>20}{'—':>9}")
            detail.append({"pair": label, "valid": a, "delta": 0.0,
                           "ci": [0.0, 0.0], "p_better": float("nan"),
                           "size_mb": sum(guess_size(x) for x in ms),
                           "members": ms, "is_champion": True})
            continue
        m = np.mean([sp[x] for x in ms], axis=0)
        d, lo, hi, prob = bootstrap_delta(sp_champ, m, sl, n=args.n)
        detail.append({"pair": label, "valid": a, "delta": d,
                       "ci": [lo, hi], "p_better": prob,
                       "size_mb": sum(guess_size(x) for x in ms), "members": ms})
        mark = ""
        if lo > 0:
            mark = "  ← 有证据更好"
        elif hi < 0:
            mark = "  ← 有证据更差"
        print(f"{label:<40}{a:>8.4f}{d:>+8.4f}"
              f"  [{lo:+.4f}, {hi:+.4f}]{prob:>9.1%}{mark}")

    # ---------- 按规则给推荐 ----------
    best = max(detail, key=lambda x: x["valid"])
    se = 0.008        # 859 study 下单组合的经验标准误
    near = [d for d in detail if best["valid"] - d["valid"] <= se]
    simple = min(near, key=lambda x: (x["size_mb"], x["valid"] * -1))
    proven = [d for d in detail if not d.get("is_champion") and d["ci"][0] > 0]

    print("\n" + "=" * 78)
    print("【推荐规则（写死，防止事后找理由）】")
    print(f"  · 第一名：{best['pair']}  valid {best['valid']:.4f}  体积 {best['size_mb']}MB")
    print(f"  · 一标准误内({se:.3f})的候选 {len(near)} 个，其中体积最小："
          f"{simple['pair']}  valid {simple['valid']:.4f}  {simple['size_mb']}MB")
    if proven:
        b = max(proven, key=lambda x: x["valid"])
        print(f"  · 唯一被 bootstrap 证明优于冠军的：{b['pair']}  "
              f"Δ {b['delta']:+.4f}  区间 [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")
        print(f"  → 建议改成：{' + '.join(b['members'])}（但必须先过 T4 耗时与 500MB）")
    else:
        print("  · 没有任何一对被证明优于现役冠军 → **保持现役组合不变**")
    print("=" * 78)
    print("提醒：体积只算权重（fp16）。加成员会线性增加前向耗时，100ms 是硬约束，")
    print("      所以最终选择 = 先看能否过 T4 实测，再看 valid。")

    if args.out:
        payload = {"singles": {n: a for n, a in singles},
                   "pairs": detail, "champion": champ,
                   "champion_valid": a_champ}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 {args.out}")


if __name__ == "__main__":
    main()
