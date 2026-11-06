# -*- coding: utf-8 -*-
"""架构自检：不训练、不下载，只验证「结构 + 参数量 + 体积 + 前向」。

为什么必须先跑这个
--------------
baseline1 有过两次"跑起来才发现结构不对"：DataParallel 下冻结骨干直接报错、
SSL 权重同名不同形状把加载打断。换骨干时这类问题只会更多，
所以新基线把自检做成独立脚本，**先过它，再谈训练**。

用法
----
    python tools/selftest_archs.py                       # 检查全部架构（不加载预训练）
    python tools/selftest_archs.py --arch dinov2_vitb14  # 只检查一个
    python tools/selftest_archs.py --arch convnext_base_in22k --weights weights/convnext_base_22k_224.pth
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archs import ALL_ARCHS, param_stats  # noqa: E402
from model3 import CXRNet  # noqa: E402


def check_one(arch, img_size, pretrained, weights, view_dim=0, pool="avg"):
    print("-" * 66)
    print(f"[{arch}] img_size={img_size} pretrained={pretrained} view_dim={view_dim}")
    try:
        model = CXRNet(arch=arch, img_size=img_size, pretrained=pretrained,
                       pool=pool, view_dim=view_dim,
                       verbose=bool(weights))
        x = torch.randn(2, 3, img_size, img_size)
        view = torch.zeros(2, view_dim) if view_dim else None
        with torch.no_grad():
            y = model(x, view)
        st = param_stats(model)
        print(f"  前向 {tuple(y.shape)} | {st['params_m']:.1f}M 参数 | "
              f"fp16 {st['fp16_mb']:.0f}MB | fp32 {st['fp32_mb']:.0f}MB")
        ok_size = st["fp16_mb"] <= 500
        print(f"  合规体积（fp16）: {'✓' if ok_size else '✗ 超 500MB'} | "
              f"impl={model.info['impl']}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ 失败: {type(e).__name__}: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default=None)
    p.add_argument("--img-size", type=int, default=392,
                   help="DINOv2 要求 14 的倍数；392=28x28 token，448=32x32")
    p.add_argument("--weights", default=None, help="本地权重文件（测试加载是否对得上）")
    p.add_argument("--view-dim", type=int, default=0)
    args = p.parse_args()

    archs = [args.arch] if args.arch else list(ALL_ARCHS)
    pretrained = args.weights if args.weights else "none"
    print("=" * 66)
    print("架构自检：只验证结构与体积，不验证精度")
    print("=" * 66)

    ok, bad = [], []
    for a in archs:
        img = args.img_size
        if a.startswith("dinov2"):
            img = (img // 14) * 14
        r = check_one(a, img, pretrained, args.weights, args.view_dim)
        (ok if r else bad).append(a)

    print("=" * 66)
    print(f"通过 {len(ok)} 个: {ok}")
    if bad:
        print(f"失败 {len(bad)} 个: {bad}")
        print("排查提示：")
        print("  · 'No module named torchvision' → 先 source venv")
        print("  · 体积超限 → 换更小的变体，或最终用 fp16 存权重")
        print("  · 权重匹配率低 → 变体不对（base/large 维度不同），或命名不兼容")


if __name__ == "__main__":
    main()
