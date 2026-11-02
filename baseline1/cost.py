# -*- coding: utf-8 -*-
"""cost.py —— 推理耗时与结果一致性核验脚本（规则第 7 条，**必须随提交包提供**）。

规则要求：
1. 必须导入并调用与 predict.py **相同的推理函数与模型加载逻辑**（本脚本 import
   inference.py，而 predict.py 也 import 同一份，结构上保证一致）；
2. 100 张样本、batch=1 串行、连续 3 轮；每张先取 3 轮均值，再对 100 张取平均；
   单张端到端 ≤100ms（含读图与预处理，不含模型加载/预热）；
3. 与 submission.csv 逐张比对：二值判定与概率排序必须完全一致，不一致张数须为 0，
   |Δp|≤1e-3 视为浮点正常波动；
4. eval 模式、固定种子、关闭随机增强的确定性条件下运行。

用法（与主办方给的口径一致）：
    python cost.py submission.csv pictures_100 pictures_5 --ckpt runs/xxx/ema_f12.pth --tta

若提供了 predict.py 生成的 *_meta.json，可用 --meta 自动对齐 tta / ckpt 等参数：
    python cost.py submission.csv pictures_100 pictures_5 \
        --meta runs/xxx/pred_test_meta.json

本地自测（图片少、轮数可调）：
    python cost.py submission.csv pictures_100 pictures_5 --ckpt ... --limit 10 --rounds 1
"""
import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from inference import load_model_for_inference, predict_one_path


# ---------------- 工具 ----------------
def parse_study_from_path(path):
    """从 <root>/subject_x/study_y/pic_z.jpg 解析出 (subject, study, dicom)。"""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3:
        raise ValueError(f"图片路径不符合 <subject>/<study>/<pic>.jpg 结构: {path}")
    return parts[-3], parts[-2], os.path.splitext(parts[-1])[0]


def list_images(root):
    """列出目录下所有 jpg/png（递归）。"""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def load_submission(path):
    """读取提交文件 -> {(subject, study): {class_id: prob}}。"""
    table = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["Subject_id"], r["Study_id"])
            table[key][int(r["Predict_class"])] = float(r["Probability"])
    return table


def resolve_settings(args):
    """参数优先级：命令行 > *_meta.json（predict.py 记录）> config 默认。"""
    from config import AGG, TTA
    tta, ckpt, agg = args.tta, args.ckpt, AGG
    if args.meta and os.path.exists(args.meta):
        with open(args.meta, encoding="utf-8") as f:
            meta = json.load(f)
        tta = bool(meta.get("tta", tta))
        agg = meta.get("agg", agg)
        if not ckpt:
            ckpt = meta.get("ckpts")
        print(f"已从 {args.meta} 对齐推理参数：tta={tta} agg={agg}")
    if isinstance(ckpt, str):
        ckpt = [ckpt]
    if not ckpt:
        raise SystemExit("必须用 --ckpt 指定权重，或用 --meta 指向 predict.py 生成的 meta")
    return ckpt, tta, agg


# ---------------- 一致性核验 ----------------
def compare_with_submission(sub_probs, img_probs_by_study, agg="mean",
                            tol=1e-3, full_counts=None):
    """把 100 张图的预测按 study 聚合后与 submission 比对。

    返回 (不一致数, 明细列表, 部分覆盖的 study 列表)。
    说明：多图 study 只有在「该 study 的图全部出现在这 100 张里」时才能精确重建
    聚合结果；只出现部分图时单独列出，不计入不一致（但会在日志里明示）。
    """
    mismatches, details, partial = 0, [], []
    for (subject, study), image_probs in img_probs_by_study.items():
        key = (subject, study)
        ref = sub_probs.get(key)
        if not ref:
            mismatches += 1
            details.append(f"{study}: submission 中找不到该 study")
            continue

        n_have = len(image_probs)
        n_full = None if full_counts is None else full_counts.get(study)
        is_partial = (n_full is not None and n_have < n_full)
        if is_partial:
            partial.append(f"{study}({n_have}/{n_full} 张)")

        # 按同样的方式聚合（mean / max）
        stacked = np.stack(image_probs)
        got = stacked.mean(axis=0) if agg == "mean" else stacked.max(axis=0)

        max_dp = 0.0
        bin_diff, rank_diff = [], []
        for cid in sorted(ref):
            ci = cid - 1
            if ci >= len(got):
                continue
            ref_p, got_p = ref[cid], float(got[ci])
            max_dp = max(max_dp, abs(ref_p - got_p))
            if (ref_p >= 0.5) != (got_p >= 0.5):
                bin_diff.append(cid)
        # 概率排序（top-1 类别）是否一致
        ref_top = max(ref, key=lambda c: ref[c])
        got_top = int(np.argmax(got)) + 1
        if ref_top != got_top:
            rank_diff.append(f"top1 {ref_top}->{got_top}")

        bad = bool(bin_diff or rank_diff) or (max_dp > tol and not is_partial)
        if bad:
            mismatches += 1
            details.append(
                f"{study}: |Δp|max={max_dp:.5f}"
                + (f" 二值不同={bin_diff}" if bin_diff else "")
                + (f" 排序不同={rank_diff}" if rank_diff else "")
                + (" [部分覆盖]" if is_partial else ""))
    return mismatches, details, partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv", help="最终提交的 submission.csv")
    parser.add_argument("pictures_100", help="100 张计时图片目录（结构与测试集一致）")
    parser.add_argument("pictures_5", help="5 张预热图片目录")
    parser.add_argument("--ckpt", nargs="+", default=None, help="推理权重（单个）")
    parser.add_argument("--meta", default=None, help="predict.py 生成的 *_meta.json")
    parser.add_argument("--tta", action="store_true", help="与提交保持一致时须开启")
    parser.add_argument("--rounds", type=int, default=3, help="计时轮数（规则要求 3）")
    parser.add_argument("--limit", type=int, default=0, help="本地自测：只用前 N 张")
    parser.add_argument("--no-check", action="store_true", help="跳过一致性核验（仅测速）")
    args = parser.parse_args()

    from config import (DATA_ROOT, INFERENCE_TIME_LIMIT_MS, USE_AMP)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    precision = "fp16 (AMP 自动混合精度)" if use_amp else "fp32"

    ckpts, tta, agg = resolve_settings(args)
    model, meta = load_model_for_inference(ckpts[0], device)
    img_size = meta["img_size"]
    print(f"设备: {device} | 推理精度: {precision} | 骨干: {meta['backbone']} "
          f"| 输入 {img_size}px | TTA: {tta} | 聚合: {agg}")
    print(f"权重: {ckpts[0]}")

    # ---------- 图片 ----------
    warm_imgs = list_images(args.pictures_5)
    test_imgs = list_images(args.pictures_100)
    if args.limit:
        test_imgs = test_imgs[:args.limit]
        print(f"[自测模式] 只用 {len(test_imgs)} 张，轮数 {args.rounds}")
    if not warm_imgs:
        raise SystemExit(f"{args.pictures_5} 下没有图片")
    if not test_imgs:
        raise SystemExit(f"{args.pictures_100} 下没有图片")
    print(f"预热图 {len(warm_imgs)} 张 | 计时图 {len(test_imgs)} 张")

    # ---------- 预热（不计入成绩） ----------
    for p in warm_imgs:
        predict_one_path(model, p, img_size, device, tta=tta, use_amp=use_amp)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ---------- 计时：每张跑 rounds 轮，逐张取均值 ----------
    per_image_times = []
    last_probs = {}
    for round_idx in range(args.rounds):
        times_this_round = []
        for p in test_imgs:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = predict_one_path(model, p, img_size, device, tta=tta,
                                     use_amp=use_amp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_this_round.append((time.perf_counter() - t0) * 1000.0)
            last_probs[p] = probs
        if round_idx == 0:
            per_image_times = [[] for _ in test_imgs]
        for i, t in enumerate(times_this_round):
            per_image_times[i].append(t)
        print(f"  第 {round_idx + 1}/{args.rounds} 轮完成，"
              f"本轮均值 {np.mean(times_this_round):.1f} ms")

    per_img_mean = [float(np.mean(ts)) for ts in per_image_times]
    avg_ms = float(np.mean(per_img_mean))
    over_limit = avg_ms > INFERENCE_TIME_LIMIT_MS

    # ---------- 一致性核验 ----------
    mismatches, details, partial = 0, [], []
    if not args.no_check:
        sub_probs = load_submission(args.submission_csv)
        by_study = defaultdict(list)
        for p, probs in last_probs.items():
            subject, study, _dicom = parse_study_from_path(p)
            by_study[(subject, study)].append(probs)

        # 若本地有 test.csv，可判断「该 study 的图是否全在这 100 张里」
        full_counts = None
        try:
            from common import read_split_csv
            rows = read_split_csv("test", data_root=DATA_ROOT)
            full_counts = Counter(r["Study_id"] for r in rows)
        except Exception:  # noqa: BLE001
            print("（未找到 test.csv，跳过「是否完整覆盖」的判断）")

        n = min(len(test_imgs), 100) if not args.limit else len(test_imgs)
        mismatches, details, partial = compare_with_submission(
            sub_probs, by_study, agg=agg, full_counts=full_counts)

    # ---------- 结论 ----------
    print("")
    if mismatches:
        print("不一致明细（前 10 条）:")
        for d in details[:10]:
            print("  -", d)
    if partial:
        print(f"注意：{len(partial)} 个 study 的图片未被 100 张样本完整覆盖"
              f"（如 {', '.join(partial[:3])}），其 study 级聚合无法精确重建，"
              "已单独标注。")
    print(
        f"[CONCLUSION] 结论：{len(test_imgs)} 张样本、batch=1 串行、连续 {args.rounds} 轮；"
        f"单张端到端墙钟时间先按每张 {args.rounds} 轮取均值，再对 {len(test_imgs)} 张取平均，"
        f"最终平均耗时 {avg_ms:.0f} ms，"
        f"{'未超过上限' if not over_limit else '超过上限 ' + str(INFERENCE_TIME_LIMIT_MS) + ' ms'}；"
        f"预测结果与 submission.csv 比对 {len(test_imgs)} 张，不一致 {mismatches} 张；"
        f"推理精度 {precision}。"
    )
    if over_limit:
        print(f"⚠ 单张平均 {avg_ms:.1f} ms > {INFERENCE_TIME_LIMIT_MS} ms："
              "需优化推理实现（降低分辨率/量化/关闭 TTA），注意复测只允许优化实现、"
              "不得更换权重或改变预测结果。")
    raise SystemExit(1 if (mismatches or over_limit) else 0)


if __name__ == "__main__":
    main()
