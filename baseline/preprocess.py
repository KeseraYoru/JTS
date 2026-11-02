# -*- coding: utf-8 -*-
"""预处理：把 46GB 的原始大图（1409~4280px）统一缩放到 600x600 小图。

为什么先预处理（Kaggle 方案同款思路）：
训练时每个 epoch 要把 3 万张图读一遍，原图平均 1.5MB，解码 + resize 非常慢；
预先缩放到 600x600（约 100KB）后，磁盘 I/O 和解码开销降一个数量级，
而且保持 subject/study/pic 目录层级不变，路径拼接逻辑完全复用。

输出：{PREP_ROOT}/{split}/{Subject_id}/{Study_id}/{Dicom_id}.jpg（RGB JPEG）

用法：
    python preprocess.py                 # 处理 train+valid+test 全部
    python preprocess.py --split train   # 只处理训练集
    python preprocess.py --limit 100     # 只处理前 100 张（调试用）
"""
import argparse
import os
from multiprocessing import Pool, cpu_count

from PIL import Image
from tqdm import tqdm

from common import read_split_csv, raw_img_path
from config import DATA_ROOT, PREP_QUALITY, PREP_ROOT, PREP_SIZE, SPLITS


def _resize_one(task):
    in_path, out_path = task
    if os.path.exists(out_path):
        return "skipped"  # 已存在则跳过，重跑不用再花半小时
    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")
            img = img.resize((PREP_SIZE, PREP_SIZE), Image.BILINEAR)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img.save(out_path, "JPEG", quality=PREP_QUALITY, optimize=True)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{in_path}: {e}"


def collect_tasks(splits, data_root, prep_root, limit=None):
    """按 CSV 列出所有 (原图路径, 输出路径)，保证不重不漏。"""
    tasks = []
    for split in splits:
        for row in read_split_csv(split, data_root):
            tasks.append((raw_img_path(row, split, data_root),
                          os.path.join(prep_root, split, row["Subject_id"],
                                       row["Study_id"], row["Dicom_id"] + ".jpg")))
    if limit:
        tasks = tasks[:limit]
    return tasks


def main():
    parser = argparse.ArgumentParser(description="大图 -> 600x600 预处理")
    parser.add_argument("--split", default="all",
                        help="train / valid / test / all（默认 all）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 张，调试用（0 = 全部）")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--prep-root", default=PREP_ROOT)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    splits = list(SPLITS) if args.split == "all" else [args.split]
    tasks = collect_tasks(splits, args.data_root, args.prep_root,
                          args.limit or None)
    print(f"待处理 {len(tasks)} 张 -> {args.prep_root}"
          f"（{PREP_SIZE}x{PREP_SIZE}, JPEG q{PREP_QUALITY}, "
          f"{args.workers} 进程）")

    errors, skipped = [], 0
    with Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(_resize_one, tasks),
                        total=len(tasks), desc="预处理"):
            if res == "skipped":
                skipped += 1
            elif res:
                errors.append(res)

    print(f"完成：{len(tasks) - skipped - len(errors)} 张新处理，"
          f"{skipped} 张已存在跳过，{len(errors)} 张失败")
    for e in errors[:20]:
        print("  失败:", e)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
