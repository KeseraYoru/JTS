# -*- coding: utf-8 -*-
"""权重平均（model soup）：把多个同架构、同尺寸的 ckpt 的 EMA 权重取平均。

为什么值得做
----------
同架构不同 seed 的模型，参数平均后往往比单个更稳（相当于在参数空间里取"共识"）。
它在 baseline1 时代测过：3 seed soup 的 valid 与最好单 seed 差距在 ±0.001，
**不提分但提稳** —— 对"提交哪一份"这种只能选一次的决定，稳比高更有价值。

严格校验（避免把不同结构平均出一堆垃圾）
------------------------------------
arch / img_size / view_dim / pool 必须完全一致；键名与形状必须一致。
任何一条不满足就拒绝执行，而不是"尽量平均"。
"""
import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--use-weight", default="ema", choices=["ema", "raw"],
                   help="平均 EMA 权重还是原始权重（默认 ema）")
    args = p.parse_args()

    ckpts = [torch.load(c, map_location="cpu", weights_only=False) for c in args.ckpt]
    keys = ["arch", "impl", "img_size", "view_dim", "pool"]
    base = {k: ckpts[0].get(k) for k in keys}
    for c, path in zip(ckpts, args.ckpt):
        for k in keys:
            if c.get(k) != base[k]:
                raise SystemExit(
                    f"结构不一致：{path} 的 {k}={c.get(k)}，而基准是 {base[k]}。\n"
                    "只能平均同架构、同输入尺寸、同池化方式、同元数据维度的权重。")

    field = "ema_state_dict" if args.use_weight == "ema" else "model_state_dict"
    states = [c.get(field) or c["model_state_dict"] for c in ckpts]
    ref = states[0]
    for s, path in zip(states[1:], args.ckpt[1:]):
        if set(s.keys()) != set(ref.keys()):
            raise SystemExit(f"{path} 的参数键与基准不一致，拒绝平均")

    avg = {}
    for k in ref:
        if not torch.is_floating_point(ref[k]):
            avg[k] = ref[k].clone()          # 整数 buffer 直接取第一份
            continue
        acc = torch.zeros_like(ref[k], dtype=torch.float32)
        for s in states:
            acc += s[k].to(torch.float32)
        avg[k] = (acc / len(states)).to(ref[k].dtype)

    # 以**第一份 ckpt 的全部元信息**为底（保留 backbone / num_classes / src_size 等
    # 各版本各自需要的字段），再覆盖两套 state_dict。
    # 只拷 arch/img_size 这几个键会导致 baseline1 找不到 "backbone" 而拒绝加载。
    out = {k: v for k, v in ckpts[0].items()
           if k not in ("model_state_dict", "ema_state_dict")}
    out.update({
        # ★ 两个键都写：baseline3 的 infer3 优先读 ema_state_dict；
        #   而 baseline1 的 inference.py 只认 model_state_dict。
        #   只写一个时，另一套代码会把整个 ckpt dict 当 state_dict 用 —— 静默加载失败。
        "ema_state_dict": avg,
        "model_state_dict": avg,
        "epoch": 0,
        "study_auc": None,
        "soup_of": args.ckpt,
        "soup_weight": args.use_weight,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"已写 {args.out}｜平均 {len(states)} 份（来源 {field}）")
    for c, p_ in zip(ckpts, args.ckpt):
        print(f"  - {p_}  (ep{c.get('epoch')}, valid {c.get('study_auc')})")


if __name__ == "__main__":
    main()
