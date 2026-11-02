# -*- coding: utf-8 -*-
"""在 valid 上扫两个模型的最佳集成权重（study 级 Macro AUC）。

背景：两个模型实力不均衡时（如 CheXFound 0.826 / EVA-X 0.81），等权平均反而可能
拉低总分——弱模型的短板类会把强模型的优势稀释掉。这个脚本在 valid 上把权重从
0 到 1 扫一遍，告诉你最好的配比（w=1.0 就是只用模型 A，w=0.0 就是只用模型 B）。

用法：
    python3 scan_weights.py --split valid \
        --inputs runs/evax_xxx/pred_valid.npz runs/chexfound_xxx/pred_valid.npz

注意：
- 只看 study 级 Macro AUC（和评测主指标一致），不做阈值优化；
- valid 只有 859 个 study，AUC 的抽样噪声约 ±0.005，所以**别在小数点后第三位纠结**，
  选一个宽的、稳定的平台区间即可；免费拿到的提升（≥0.005）值得用，低于噪声的差异不值得。
"""
import argparse

import numpy as np

from common import (aggregate_by_study, encode_labels, format_auc_report,
                    macro_auc, per_class_auc, read_split_csv)


def load_probs(path, rows):
    """读 npz 并校验图像顺序与 CSV 一致。"""
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["dicom_ids"]]
    if ids != [r["Dicom_id"] for r in rows]:
        raise SystemExit(f"{path} 的图像顺序与 CSV 不一致，请确认用的是同一个 split")
    return data["probs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid", choices=["valid"])
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="两个模型的 pred_valid.npz（前者为 A，权重 w）")
    parser.add_argument("--agg", default="mean", choices=["mean", "max"])
    parser.add_argument("--step", type=float, default=0.05,
                        help="权重扫描步长（默认 0.05）")
    args = parser.parse_args()

    if len(args.inputs) != 2:
        raise SystemExit("目前只支持 2 个模型（对第一个扫权重，第二个取 1-w）")

    rows = read_split_csv(args.split)
    labels = encode_labels(rows)
    probs_a = load_probs(args.inputs[0], rows)
    probs_b = load_probs(args.inputs[1], rows)

    # 标签按 study 聚合一次即可（与概率无关）
    _, _, study_labels = aggregate_by_study(rows, probs_a, labels=labels,
                                            how=args.agg)

    print(f"模型 A: {args.inputs[0]}")
    print(f"模型 B: {args.inputs[1]}")
    print(f"聚合: {args.agg} | 扫描步长: {args.step}")
    print("")
    print(f"{'w_A':>5} {'w_B':>5}  study_AUC   备注")
    print("-" * 46)

    best_auc, best_w = -1.0, None
    w = 0.0
    while w <= 1.0 + 1e-9:
        blend = w * probs_a + (1.0 - w) * probs_b
        _, study_probs, _ = aggregate_by_study(rows, blend, how=args.agg)
        auc = macro_auc(study_labels, study_probs)
        note = ""
        if w == 0.0:
            note = "= 只用 B"
        elif abs(w - 1.0) < 1e-9:
            note = "= 只用 A"
        elif abs(w - 0.5) < 1e-9:
            note = "= 等权"
        if auc > best_auc:
            best_auc, best_w = auc, w
        print(f"{w:5.2f} {1.0 - w:5.2f}  {auc:.4f}   {note}")
        w = round(w + args.step, 10)

    blend = best_w * probs_a + (1.0 - best_w) * probs_b
    _, study_probs, _ = aggregate_by_study(rows, blend, how=args.agg)
    aucs = per_class_auc(study_labels, study_probs)

    print("-" * 46)
    print(f"最优: w_A={best_w:.2f}  w_B={1.0 - best_w:.2f}  "
          f"study AUC {best_auc:.4f}")
    print("  每类 AUC:", format_auc_report(aucs))
    print("")
    print("提示：859 个 study 的 AUC 噪声约 ±0.005，优先选稳定的平台区间，"
          "别追小数点后第三位。")
    print(f"用它出提交：\n  python3 ensemble.py --split test "
          f"--inputs {args.inputs[0].replace('valid', 'test')} "
          f"{args.inputs[1].replace('valid', 'test')} "
          f"--weights {best_w:.2f} {1.0 - best_w:.2f} --out submission.csv")


if __name__ == "__main__":
    main()
