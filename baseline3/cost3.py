# -*- coding: utf-8 -*-
"""cost3.py —— 推理耗时与一致性核验（规则第 7 条，**必须随提交包提供**）。

口径（照抄规则，不要自己发明）
--------------------------
- 100 张样本、batch=1 串行、连续 3 轮；
- 每张先取 3 轮均值，再对 100 张取平均；单张端到端 ≤100ms（含读图与预处理，
  不含模型加载与预热）；
- 与 submission.csv 逐张比对：二值判定与概率排序必须完全一致，不一致须为 0 张，
  |Δp|≤1e-3 视为浮点正常波动。

与 baseline1 的三处差异
--------------------
1. 多模型**共享一次读图与预处理**（否则同一张原图被解码 N 遍，耗时严重高估）；
2. 支持 logit 融合与 Normal Gating —— 但必须与 ensemble3 的生成参数完全一致，
   所以推荐用 `--meta` 直接读 predict3 记录的参数；
3. ckpt 的 arch/img_size/view_dim 全部从文件里读，不靠命令行猜。

用法
----
    python cost3.py submission.csv pictures_100 pictures_5 \
        --ckpt runs/full_dino/best.pth --meta runs/pred_test_dino_meta.json
"""
import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from common3 import apply_normal_gating, read_split_csv
from config3 import INFERENCE_TIME_LIMIT_MS, USE_AMP
from infer3 import load_from_ckpt, predict_one_multi


def parse_study_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"图片路径不符合 <subject>/<study>/<pic>.jpg 结构: {path}")
    return parts[-3], parts[-2], os.path.splitext(parts[-1])[0]


def list_images(root):
    out = []
    for dirpath, _d, files in os.walk(root):
        for fn in sorted(files):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def load_submission(path):
    table = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            table[(r["Subject_id"], r["Study_id"])][int(r["Predict_class"])] = float(r["Probability"])
    return table


def compare(sub_probs, by_study, agg="mean", gate=0.0, tol=1e-3, full_counts=None):
    mismatches, details, partial = 0, [], []
    for (subject, study), plist in by_study.items():
        ref = sub_probs.get((subject, study))
        if not ref:
            mismatches += 1
            details.append(f"{study}: submission 中找不到该 study")
            continue
        n_have = len(plist)
        n_full = None if full_counts is None else full_counts.get(study)
        is_partial = (n_full is not None and n_have < n_full)
        if is_partial:
            partial.append(f"{study}({n_have}/{n_full} 张)")

        stacked = np.stack(plist)
        got = stacked.mean(axis=0) if agg == "mean" else stacked.max(axis=0)
        if gate > 0:
            got = apply_normal_gating(got[None, :], gate)[0]

        max_dp, bin_diff, rank_diff = 0.0, [], []
        for cid in sorted(ref):
            ci = cid - 1
            if ci >= len(got):
                continue
            ref_p, got_p = ref[cid], float(got[ci])
            max_dp = max(max_dp, abs(ref_p - got_p))
            if (ref_p >= 0.5) != (got_p >= 0.5):
                bin_diff.append(cid)
        ref_top = max(ref, key=lambda c: ref[c])
        got_top = int(np.argmax(got)) + 1
        if ref_top != got_top:
            rank_diff.append(f"top1 {ref_top}->{got_top}")

        bad = bool(bin_diff or rank_diff) or (max_dp > tol and not is_partial)
        if bad:
            mismatches += 1
            details.append(f"{study}: |Δp|max={max_dp:.5f}"
                           + (f" 二值不同={bin_diff}" if bin_diff else "")
                           + (f" 排序不同={rank_diff}" if rank_diff else "")
                           + (" [部分覆盖]" if is_partial else ""))
    return mismatches, details, partial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission_csv")
    p.add_argument("pictures_100")
    p.add_argument("pictures_5")
    p.add_argument("--ckpt", nargs="+", default=None)
    p.add_argument("--meta", default=None, help="predict3.py 生成的 *_meta.json")
    p.add_argument("--fusion", default="mean", choices=["mean", "logit"])
    p.add_argument("--gate", type=float, default=0.0)
    p.add_argument("--agg", default="mean", choices=["mean", "max"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--img-size", type=int, default=0)
    p.add_argument("--disable-ema", action="store_true")
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    precision = "fp16 (AMP 自动混合精度)" if use_amp else "fp32"

    ckpts, tta, agg = args.ckpt, "none", args.agg
    if args.meta and os.path.exists(args.meta):
        with open(args.meta, encoding="utf-8") as f:
            m = json.load(f)
        ckpts = ckpts or m.get("ckpts")
        agg = m.get("agg", agg)
        print(f"已从 {args.meta} 对齐推理参数：img_size={m.get('img_size')} "
              f"src_size={m.get('src_size')}")
    if not ckpts:
        raise SystemExit("必须用 --ckpt 指定权重，或用 --meta 指向 predict3 的 meta")

    models, metas = [], []
    for cp in ckpts:
        model, meta = load_from_ckpt(cp, device, use_ema=not args.disable_ema,
                                     img_size_override=args.img_size or None)
        models.append(model)
        metas.append(meta)
    img_size = metas[0]["img_size"]
    src_size = metas[0]["src_size"]
    if len({m["img_size"] for m in metas}) > 1:
        raise SystemExit("多个 ckpt 的 img_size 不一致，无法共享预处理")
    if len({m["view_dim"] for m in metas}) > 1:
        raise SystemExit("多个 ckpt 的 view_dim 不一致，无法共享输入")

    print(f"设备: {device} | 推理精度: {precision} | 骨干: "
          f"{' + '.join(m['arch'] for m in metas)} | 输入 {img_size}px | "
          f"draft {src_size}px | 融合: {args.fusion} | 聚合: {agg}"
          + (f" | gating {args.gate}" if args.gate else ""))
    for cp in ckpts:
        print(f"  - {cp}")

    # ViewPosition 需要按 Dicom_id 查 test.csv；查不到就用全零（等价于"未知"）
    dicom_lut = {}
    try:
        for r in read_split_csv("test"):
            dicom_lut[r["Dicom_id"]] = r
    except Exception as e:  # noqa: BLE001
        print(f"（未读到 test.csv，ViewPosition 全部按未知处理：{e}）")

    def row_for(path):
        _s, _st, dic = parse_study_from_path(path)
        return dicom_lut.get(dic, {"Dicom_id": dic, "Subject_id": _s, "Study_id": _st,
                                   "ViewPosition": ""})

    warm_imgs = list_images(args.pictures_5)
    test_imgs = list_images(args.pictures_100)
    if args.limit:
        test_imgs = test_imgs[:args.limit]
        print(f"[自测模式] 只用 {len(test_imgs)} 张，轮数 {args.rounds}")
    if not warm_imgs or not test_imgs:
        raise SystemExit("pictures_5 / pictures_100 下没有图片")
    print(f"预热图 {len(warm_imgs)} 张 | 计时图 {len(test_imgs)} 张")

    for path in warm_imgs:
        predict_one_multi(models, path, row_for(path), img_size, device,
                          use_amp=use_amp, src_size=src_size, method=args.fusion)
    if device.type == "cuda":
        torch.cuda.synchronize()

    per_image_times, last_probs = [], {}
    for r_i in range(args.rounds):
        times = []
        for path in test_imgs:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = predict_one_multi(models, path, row_for(path), img_size, device,
                                      use_amp=use_amp, src_size=src_size,
                                      method=args.fusion)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
            last_probs[path] = probs
        if r_i == 0:
            per_image_times = [[] for _ in test_imgs]
        for i, t in enumerate(times):
            per_image_times[i].append(t)
        print(f"  第 {r_i + 1}/{args.rounds} 轮完成，本轮均值 {np.mean(times):.1f} ms")

    avg_ms = float(np.mean([np.mean(ts) for ts in per_image_times]))
    over = avg_ms > INFERENCE_TIME_LIMIT_MS

    mismatches, details, partial = 0, [], []
    if not args.no_check:
        sub = load_submission(args.submission_csv)
        by_study = defaultdict(list)
        for path, probs in last_probs.items():
            s, st, _d = parse_study_from_path(path)
            by_study[(s, st)].append(probs)
        full_counts = None
        try:
            rows = read_split_csv("test")
            full_counts = Counter(r["Study_id"] for r in rows)
        except Exception:  # noqa: BLE001
            print("（未找到 test.csv，跳过「是否完整覆盖」判断）")
        mismatches, details, partial = compare(sub, by_study, agg=agg, gate=args.gate,
                                               full_counts=full_counts)

    print("")
    if mismatches:
        print("不一致明细（前 10 条）:")
        for d in details[:10]:
            print("  -", d)
    if partial:
        print(f"注意：{len(partial)} 个 study 的图未被 100 张样本完整覆盖"
              f"（如 {', '.join(partial[:3])}），已单独标注。")
    print(f"[CONCLUSION] 结论：{len(test_imgs)} 张样本、batch=1 串行、连续 {args.rounds} 轮；"
          f"单张端到端墙钟时间先按每张 {args.rounds} 轮取均值，再对 {len(test_imgs)} 张取平均，"
          f"最终平均耗时 {avg_ms:.0f} ms，"
          f"{'未超过上限' if not over else '超过上限 ' + str(INFERENCE_TIME_LIMIT_MS) + ' ms'}；"
          f"预测结果与 submission.csv 比对 {len(test_imgs)} 张，不一致 {mismatches} 张；"
          f"推理精度 {precision}。")
    if over:
        print(f"⚠ 单张平均 {avg_ms:.1f} ms > {INFERENCE_TIME_LIMIT_MS} ms："
              "复测只允许优化实现（精度/算子/量化/编译），不得更换权重或改变预测结果。")
    raise SystemExit(1 if (mismatches or over) else 0)


if __name__ == "__main__":
    main()
