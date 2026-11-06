# -*- coding: utf-8 -*-
"""三档分辨率计时：拆开测「端到端」与「纯前向」，用于判断 T4 上能不能过 100ms。

为什么必须拆开（2026-11-02 修正）
--------------------------------
早期版本把总耗时整体乘 3.8 当作 T4 估算 —— **这是错的**，会严重高估。
实测（5090，convnext_base，fp16，batch=1，原图 + draft）：
    384px 端到端 26.8ms / 448px 26.8ms / 512px 27.2ms
三个尺寸几乎一样 → 说明「前向」只占不到 1ms，**瓶颈全在 CPU 侧的解码与 resize**。
（convnext_base@512 batch=1 只有 78 GFLOPs，5090 用量极低。）

正确估算：
    T4 ≈ CPU 部分 × cpu_factor  +  前向 × t4_factor
    其中 CPU 部分 = 端到端 − 纯前向

为什么这个工具要存在：
- cost.py 的输入尺寸来自 ckpt 的 img_size，想测 448/512 得先有对应模型；
  但卷积骨干权重与输入尺寸无关，拿 384 训好的权重就能测它的耗时。
- 一致性核对仍必须用 cost.py（那条路要跟 submission.csv 比对），本工具只管时间。

用法：
    python tools/bench_timing.py                        # 随机权重，测 384/448/512
    python tools/bench_timing.py --ckpt runs/soup.pth   # 用真实权重
    python tools/bench_timing.py --sizes 384 512 640 --n 100
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

# 允许从 baseline1/ 根目录直接运行（python tools/bench_timing.py）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import raw_img_path, read_split_csv          # noqa: E402
from config import DATA_ROOT, USE_AMP                     # noqa: E402
from inference import (DEFAULT_SRC_SIZE, load_model_for_inference,  # noqa: E402
                       predict_one_path)
from models import build_model                           # noqa: E402


def time_forward_only(model, size, device, use_amp, iters=50, warmup=10):
    """纯前向：张量预先准备好，只测 model(x)，不含任何读图/解码。"""
    x = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[384, 448, 512])
    p.add_argument("--backbone", default="convnext_base")
    p.add_argument("--ckpt", default=None, help="给了就用真实权重，否则用随机权重（计时等价）")
    p.add_argument("--n", type=int, default=100, help="端到端计时图数量（规则口径是 100）")
    p.add_argument("--warmup", type=int, default=5, help="预热张数（不计入）")
    p.add_argument("--src-size", type=int, default=DEFAULT_SRC_SIZE,
                   help="draft 解码尺度，与训练用的源图边长一致（data600 → 600）")
    p.add_argument("--t4-factor", type=float, default=3.8,
                   help="T4 相对本机的**纯前向**折算系数（fp16 算力约 1/3.5~1/4）")
    p.add_argument("--cpu-factor", type=float, default=1.5,
                   help="T4 容器 CPU 相对本机 CPU 的慢速系数（8 核，保守取 1.5）")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    rows = read_split_csv("test")[:args.n + args.warmup]
    paths = [raw_img_path(r, "test", DATA_ROOT) for r in rows]
    print(f"计时图 {len(paths)} 张（原图，{os.path.getsize(paths[0]) / 1e6:.2f} MB/张）"
          f" | 设备 {device} | AMP {use_amp} | draft {args.src_size}px")
    if torch.cuda.device_count() > 1:
        print(f"注意：本脚本只用 device 0（核验口径是单卡 batch=1 串行）")

    header = (f"{'尺寸':>6}{'端到端':>10}{'纯前向':>10}{'CPU部分':>10}"
              f"{'T4估算':>12}{'判定':>14}")
    print("\n" + header)
    print("-" * len(header))

    for size in args.sizes:
        if args.ckpt:
            model, _ = load_model_for_inference(args.ckpt, device, img_size=size)
        else:
            model = build_model(args.backbone, pretrained=False).eval().to(device)

        # ① 纯前向
        fwd = time_forward_only(model, size, device, use_amp)

        # ② 端到端（与 cost.py 同一条路：读原图 → draft → 变换 → 前向）
        for pth in paths[:args.warmup]:
            predict_one_path(model, pth, size, device, tta=False,
                             use_amp=use_amp, src_size=args.src_size)
        if device.type == "cuda":
            torch.cuda.synchronize()

        ts = []
        for pth in paths[args.warmup:]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict_one_path(model, pth, size, device, tta=False,
                             use_amp=use_amp, src_size=args.src_size)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        e2e = float(ts.mean())
        cpu_part = max(0.0, e2e - fwd)
        t4 = cpu_part * args.cpu_factor + fwd * args.t4_factor

        if t4 < 76:
            verdict = "✓ 安全"
        elif t4 < 95:
            verdict = "△ 贴线"
        else:
            verdict = "✗ 超限风险"
        print(f"{size:>6}{e2e:>9.1f}ms{fwd:>9.2f}ms{cpu_part:>9.1f}ms"
              f"{t4:>10.0f}ms{verdict:>14}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nT4 估算公式：CPU部分 × {args.cpu_factor}（T4 容器 8 核）"
          f" + 纯前向 × {args.t4_factor}（fp16 算力比）")
    print("判据：< 76ms 安全（留 25% 余量）；76~95ms 贴线；> 95ms 回退。")
    print("注意：本机 CPU 若远快于 T4 容器，cpu_factor 应上调（保守取 1.5~2.0）。")


if __name__ == "__main__":
    main()
