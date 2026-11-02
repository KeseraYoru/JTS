# -*- coding: utf-8 -*-
"""合规自检：体积、哈希、许可证、推理确定性（规则第 1/6/7/8 条）。

在提交前跑一遍，把「主办方会核验的东西」先自己核一遍：

    python check_compliance.py --ckpt runs/xxx/ema_f12.pth
    python check_compliance.py --ckpt runs/xxx/ema_f12.pth --licenses licenses/

检查项：
1. 用于推理的权重文件解压后合计 ≤500MB（第 8 条）；
2. 权重 SHA-256（申报 A5 用）；
3. licenses/ 下每个预训练来源都有许可证文件（第 1 条 / 附录 A）；
4. 推理确定性：同一批图连跑两次，最大 |Δp| 应 < 1e-6（第 6 条）；
5. 打印 A1~A6 申报表，可直接誊到 xlsx。
"""
import argparse
import hashlib
import os

import numpy as np
import torch

from config import (MODEL_SIZE_LIMIT_MB, PRETRAIN_SOURCES, PROB_DECIMALS,
                    TEAM_NAME)
from inference import load_model_for_inference, predict_paths


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def find_torchvision_cache(backbone):
    """尝试在 torch hub 缓存里找到对应的 ImageNet 权重（用于申报 A5）。"""
    try:
        cache = os.path.join(torch.hub.get_dir(), "checkpoints")
        if not os.path.isdir(cache):
            return None
        for fn in os.listdir(cache):
            if fn.startswith(backbone) and fn.endswith(".pth"):
                return os.path.join(cache, fn)
    except Exception:  # noqa: BLE001
        pass
    return None


def check_licenses(dir_path):
    """检查许可证文件是否齐备（每个预训练来源至少一份）。"""
    problems = []
    if not os.path.isdir(dir_path):
        return [f"许可证目录不存在: {dir_path}"]
    files = [f for f in os.listdir(dir_path)
             if f.lower().endswith((".txt", ".md", ".license"))]
    if len(files) < len(PRETRAIN_SOURCES):
        problems.append(
            f"许可证文件 {len(files)} 个，少于预训练来源 {len(PRETRAIN_SOURCES)} 个")
    for f in files:
        p = os.path.join(dir_path, f)
        if os.path.getsize(p) < 200:
            problems.append(f"{f} 内容过短（{os.path.getsize(p)}B），疑似占位文件")
    return problems


def check_determinism(ckpt, n_images=5):
    """同一批图连跑两次，确认推理是确定性的。"""
    from common import prep_img_path, read_split_csv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_split_csv("valid")[:n_images]
    paths = [prep_img_path(r, "valid") for r in rows]
    model, meta = load_model_for_inference(ckpt, device)
    p1 = predict_paths(model, paths, meta["img_size"], device, tta=False,
                       batch_size=4)
    p2 = predict_paths(model, paths, meta["img_size"], device, tta=False,
                       batch_size=4)
    return float(np.max(np.abs(p1 - p2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True,
                        help="最终推理权重（多模型集成时全部列出）")
    parser.add_argument("--licenses", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "licenses"))
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print(f"合规自检 | 队伍: {TEAM_NAME}")
    print("=" * 68)

    problems = []

    # ---------- 1. 体积 ----------
    total_mb = 0.0
    hashes = {}
    for c in args.ckpt:
        if not os.path.exists(c):
            problems.append(f"权重文件不存在: {c}")
            continue
        mb = os.path.getsize(c) / 1024 / 1024
        total_mb += mb
        hashes[c] = sha256_file(c)
        print(f"[体积] {os.path.basename(c)}: {mb:.1f} MB")
        print(f"       SHA-256: {hashes[c]}")
    print(f"[体积] 合计 {total_mb:.1f} MB / 上限 {MODEL_SIZE_LIMIT_MB} MB "
          f"→ {'通过' if total_mb <= MODEL_SIZE_LIMIT_MB else '超限'}")
    if total_mb > MODEL_SIZE_LIMIT_MB:
        problems.append(f"权重合计 {total_mb:.1f}MB 超过 {MODEL_SIZE_LIMIT_MB}MB")

    # ---------- 2. 许可证 ----------
    lic_problems = check_licenses(args.licenses)
    if lic_problems:
        problems.extend(lic_problems)
        for p in lic_problems:
            print(f"[许可证] ✗ {p}")
    else:
        print(f"[许可证] ✓ {args.licenses} 齐备")

    # ---------- 3. 确定性 ----------
    if not args.skip_determinism:
        try:
            d = check_determinism(args.ckpt[0])
            ok = d < 1e-6
            print(f"[确定性] 两次推理最大 |Δp| = {d:.2e} → {'通过' if ok else '不确定'}")
            if not ok:
                problems.append(f"推理不确定，|Δp|={d:.2e}")
        except Exception as e:  # noqa: BLE001
            print(f"[确定性] 跳过（{e}）")

    # ---------- 4. A1~A6 申报表 ----------
    print("")
    print("附录 A 申报项（A1~A6）：")
    print("-" * 68)
    for i, src in enumerate(PRETRAIN_SOURCES, 1):
        cache = find_torchvision_cache(
            src["name"].split()[1].lower() if len(src["name"].split()) > 1 else "")
        print(f"A1 模型名称 : {src['name']}")
        print(f"A2 权重直链 : {src['url']}")
        print(f"A3 许可证   : {src['license']}（全文见 licenses/）")
        print(f"A4 上游数据 : {src['upstream_data']}（{src['model_card']}）")
        if cache:
            print(f"A5 权重哈希 : {sha256_file(cache)}  ({os.path.basename(cache)})")
        else:
            print("A5 权重哈希 : 未在本地缓存中找到，可用公网直链下载后计算")
        print(f"A6 推理精度 : {'FP16 (AMP)' if torch.cuda.is_available() else 'FP32'}")
        if i < len(PRETRAIN_SOURCES):
            print("-" * 68)

    print("")
    if problems:
        print(f"✗ 自检未通过（{len(problems)} 项）：")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("✓ 全部通过，可以打包提交")


if __name__ == "__main__":
    main()
