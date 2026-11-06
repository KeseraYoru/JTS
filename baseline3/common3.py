# -*- coding: utf-8 -*-
"""公共函数：CSV、标签编码、AUC、study 聚合、提交写出、后处理。

与 baseline1/common.py 的关系：AUC / 聚合 / 提交这三块逻辑已被实测验证过，原样沿用；
本文件新增三样 baseline1 没有的东西：
1. `encode_view()`：ViewPosition 元数据编码（正位/侧位），此前完全没被利用；
2. `apply_normal_gating()`：类 10 与其余 9 类互斥的先验后处理（同类赛事冠军的做法）；
3. `rank_fuse()`：秩平均融合（皮肤病 Kaggle 第一名用的正是它，而非概率平均）。
"""
import csv
import os

import numpy as np

from config3 import CLASS_IDS, DATA_ROOT, NUM_CLASSES, PREP_ROOT, SPLITS, VIEW_DIM, VIEW_KEYS


# ---------------- CSV ----------------
def _read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 CSV: {path}\n"
            "训练机常见原因：只传了图片没传标注，或 CXR_DATA_ROOT 指错了目录。")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, str):
                r[k] = v.strip()
    return rows


def read_split_csv(split, data_root=None):
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到 {split}")
    return _read_csv(os.path.join(data_root or DATA_ROOT, f"{split}.csv"))


def read_csv_path(path):
    return _read_csv(path)


def raw_img_path(row, split, data_root=None):
    return os.path.join(data_root or DATA_ROOT, split, row["Subject_id"],
                        row["Study_id"], row["Dicom_id"] + ".jpg")


def prep_img_path(row, split, prep_root=None):
    if not split:
        raise ValueError("split 不能为空（train/valid/test）")
    return os.path.join(prep_root or PREP_ROOT, split, row["Subject_id"],
                        row["Study_id"], row["Dicom_id"] + ".jpg")


# ---------------- 标签 ----------------
def parse_predict_class(s):
    v = np.zeros(NUM_CLASSES, dtype=np.float32)
    if s is None:
        return v
    for tok in str(s).split(";"):
        tok = tok.strip()
        if not tok or tok.lower() == "nan":
            continue
        cid = int(tok)
        if cid in CLASS_IDS:
            v[CLASS_IDS.index(cid)] = 1.0
        else:
            raise ValueError(f"未知类别 id: {cid}")
    return v


def encode_labels(rows):
    return np.stack([parse_predict_class(r.get("Predict_class", "")) for r in rows])


def encode_view(rows):
    """ViewPosition -> one-hot（4 维：PA / AP / Lateral / 未知）。

    1202 行该字段为空，全部落到"未知"，不会报错。
    """
    out = np.zeros((len(rows), VIEW_DIM), dtype=np.float32)
    for i, r in enumerate(rows):
        v = (r.get("ViewPosition") or "").strip().upper()
        out[i, VIEW_KEYS.get(v, VIEW_DIM - 1)] = 1.0
    return out


# ---------------- AUC ----------------
def _binary_auc(y, s):
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_ranks_pos = ranks[y[order] == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def per_class_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return [_binary_auc(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])]


def macro_auc(y_true, y_score):
    aucs = [a for a in per_class_auc(y_true, y_score) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def format_auc_report(aucs, prefix=""):
    parts = [f"{cid:>2}:{a:.4f}" if not np.isnan(a) else f"{cid:>2}: nan"
             for cid, a in zip(CLASS_IDS, aucs)]
    return prefix + " ".join(parts)


# ---------------- study 聚合 ----------------
def aggregate_by_study(rows, probs, how="mean", labels=None):
    """图像级 -> study 级。多图确定性聚合属「患者平滑」，规则补充七明确允许。"""
    if how not in ("mean", "max"):
        raise ValueError(f"不支持的聚合方式: {how}")
    order, groups = [], {}
    for i, r in enumerate(rows):
        sid = r["Study_id"]
        if sid not in groups:
            groups[sid] = {"row": r, "idx": []}
            order.append(sid)
        groups[sid]["idx"].append(i)

    study_rows, study_probs = [], []
    for sid in order:
        idx = np.asarray(groups[sid]["idx"])
        p = probs[idx]
        study_rows.append(groups[sid]["row"])
        study_probs.append(p.mean(axis=0) if how == "mean" else p.max(axis=0))
    study_probs = np.asarray(study_probs, dtype=np.float64)

    study_labels = None
    if labels is not None:
        study_labels = np.stack([labels[groups[sid]["idx"][0]] for sid in order])
    return study_rows, study_probs, study_labels


# ---------------- 后处理 ----------------
def apply_normal_gating(probs, gamma, normal_index=NUM_CLASSES - 1):
    """Normal Gating：用「无异常」的概率做门，压低其余 9 类。

    probs: [N, 10]（建议传 study 级概率）
    gamma: 0 关闭；1.0 表示 p_abn *= (1 - p_normal)；>1 更激进

    为什么这对 AUC 有效：类 10 与其余 9 类在数据里互斥（12649 行实证）。
    模型对"完全正常"越确信，异常类就越不该排在前面；乘上 (1-p_normal) 会把
    假阳性的排序压下去，真阳性相对上升 —— 这是确定性的、不拟合参数的后处理，
    性质同 TTA/集成，规则允许。
    """
    if not gamma or gamma <= 0:
        return probs
    p = np.asarray(probs, dtype=np.float64).copy()
    gate = np.clip(p[:, normal_index:normal_index + 1], 0.0, 1.0) ** gamma
    p[:, :normal_index] = p[:, :normal_index] * (1.0 - gate)
    return p


def rank_fuse(mats, weights=None):
    """秩平均融合：每列转成 0~1 的秩再平均。

    与概率平均的差别：概率平均会被"说话更满"的模型主导（一个输出 0.99 的模型
    会压过两个输出 0.6 的模型）；秩平均只看顺序，天然消除校准尺度差异。
    AUC 只吃排序，所以这一招不增加任何推理成本，只改融合方式。
    """
    mats = [np.asarray(m, dtype=np.float64) for m in mats]
    n, c = mats[0].shape
    if weights is None:
        weights = [1.0] * len(mats)
    acc = np.zeros((n, c), dtype=np.float64)
    wsum = 0.0
    for m, w in zip(mats, weights):
        o = np.argsort(m, axis=0, kind="mergesort")
        r = np.empty(m.shape, dtype=np.float64)
        # 用平均秩处理并列，避免"同一概率被排成不同顺序"引入噪声
        for j in range(c):
            col = m[:, j]
            order = o[:, j]
            sorted_vals = col[order]
            ranks = np.empty(n, dtype=np.float64)
            i = 0
            while i < n:
                k = i
                while k + 1 < n and sorted_vals[k + 1] == sorted_vals[i]:
                    k += 1
                ranks[i:k + 1] = (i + k) / 2.0
                i = k + 1
            r[order, j] = ranks
        acc += w * (r / max(n - 1, 1))
        wsum += w
    return acc / wsum


def logit_fuse(mats, weights=None):
    """logit 平均：先转 log-odds 再平均，等价于几何平均的排序版本。"""
    mats = [np.clip(np.asarray(m, dtype=np.float64), 1e-6, 1 - 1e-6) for m in mats]
    if weights is None:
        weights = [1.0] * len(mats)
    wsum = sum(weights)
    acc = np.zeros_like(mats[0])
    for m, w in zip(mats, weights):
        acc += w * (np.log(m) - np.log1p(-m))
    return 1.0 / (1.0 + np.exp(-acc / wsum))


# ---------------- 提交 ----------------
def write_submission(study_rows, study_probs, out_path, all_classes=True,
                     threshold=0.5, decimals=6):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n_rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Subject_id", "Study_id", "Predict_class", "Probability"])
        for meta, probs in zip(study_rows, study_probs):
            for ci, cid in enumerate(CLASS_IDS):
                p = float(probs[ci])
                if not all_classes and p < threshold:
                    continue
                w.writerow([meta["Subject_id"], meta["Study_id"], cid, f"{p:.{decimals}f}"])
                n_rows += 1
    return n_rows


def check_submission(out_path, test_rows, all_classes=True):
    errors = []
    with open(out_path, newline="", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    if not rd:
        return ["submission.csv 为空"]
    n_study = len({r["Study_id"] for r in test_rows})
    if all_classes and len(rd) != n_study * NUM_CLASSES:
        errors.append(f"行数 {len(rd)} != study 数 {n_study} x {NUM_CLASSES}")
    got = {r["Study_id"] for r in rd}
    want = {r["Study_id"] for r in test_rows}
    if got != want:
        errors.append(f"study 集合不一致：多 {len(got - want)}，少 {len(want - got)}")
    bad_p = [r for r in rd if not (0.0 <= float(r["Probability"]) <= 1.0)]
    if bad_p:
        errors.append(f"{len(bad_p)} 行概率超出 [0,1]")
    bad_c = {r["Predict_class"] for r in rd} - {str(c) for c in CLASS_IDS}
    if bad_c:
        errors.append(f"出现未知类别 id: {sorted(bad_c)}")
    return errors
