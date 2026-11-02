# -*- coding: utf-8 -*-
"""公共函数：CSV 读取、标签多热编码、路径拼接、AUC 计算、study 级聚合、提交文件生成。

与模型无关，因此从 baseline/ 原样沿用（这部分逻辑已经过实测验证）：
- 只依赖标准库 + numpy；
- Study_id / Subject_id 一律按字符串处理，防止前导零丢失导致提交对不上。
"""
import csv
import os

import numpy as np

from config import CLASS_IDS, DATA_ROOT, NUM_CLASSES, PREP_ROOT, SPLITS


# ---------------- CSV 读取 ----------------
def read_split_csv(split, data_root=None):
    """读取某个 split 的 CSV，返回 list[dict]（字段一律按字符串读）。"""
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到: {split}")
    path = os.path.join(data_root or DATA_ROOT, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到数据文件: {path}\n"
            "常见原因：\n"
            "① 在 GPU 服务器上跑但没传 CSV——图片不用传，但这 3 个标注文件要传：\n"
            "   rsync -av cxr_first/train.csv cxr_first/valid.csv cxr_first/test.csv "
            "用户名@服务器:~/cxr/cxr_first/\n"
            "② 数据在别的路径——设环境变量 CXR_DATA_ROOT 指向数据目录再跑。"
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, str):
                r[k] = v.strip()
    return rows


def read_csv_path(path):
    """读取任意 CSV（K 折的 fold 文件用），返回 list[dict]，字段一律按字符串。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 CSV: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, str):
                r[k] = v.strip()
    return rows


def raw_img_path(row, split, data_root=None):
    """原始大图路径：{data_root}/{split}/{Subject_id}/{Study_id}/{Dicom_id}.jpg"""
    return os.path.join(data_root or DATA_ROOT, split, row["Subject_id"],
                        row["Study_id"], row["Dicom_id"] + ".jpg")


def prep_img_path(row, split, prep_root=None):
    """预处理后小图路径：{prep_root}/{split}/{Subject_id}/{Study_id}/{Dicom_id}.jpg"""
    if not split:
        raise ValueError(
            "split 不能为空（应为 train / valid / test）——"
            "检查调用方传给 ChestXrayDataset 的 split 参数")
    return os.path.join(prep_root or PREP_ROOT, split, row["Subject_id"],
                        row["Study_id"], row["Dicom_id"] + ".jpg")


# ---------------- 标签编码 ----------------
def parse_predict_class(s):
    """'7;9' -> 10 维 0/1 向量；空串/NaN -> 全 0。"""
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
            raise ValueError(f"未知类别 id: {cid}（应在 1~10 之间）")
    return v


def encode_labels(rows):
    """把整个 split 的 Predict_class 编码成 [N, 10] 的 0/1 矩阵。"""
    return np.stack([parse_predict_class(r.get("Predict_class", "")) for r in rows])


# ---------------- AUC ----------------
def _binary_auc(y, s):
    """单个类别的 AUC（基于秩，天然处理并列）。正负样本缺一类时返回 nan。"""
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
    while i < len(s):  # 并列的预测值取平均秩
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_ranks_pos = ranks[y[order] == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def per_class_auc(y_true, y_score):
    """返回长度 10 的每类 AUC（缺正/负样本的类为 nan）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return [_binary_auc(y_true[:, i], y_score[:, i])
            for i in range(y_true.shape[1])]


def macro_auc(y_true, y_score):
    """宏平均 AUC：十类等权平均（比赛主指标）。"""
    aucs = [a for a in per_class_auc(y_true, y_score) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def format_auc_report(aucs, prefix=""):
    """把每类 AUC 格式化成一行日志，方便按类定位薄弱项。"""
    parts = []
    for cid, a in zip(CLASS_IDS, aucs):
        parts.append(f"{cid:>2}:{a:.4f}" if not np.isnan(a) else f"{cid:>2}: nan")
    return prefix + " ".join(parts)


# ---------------- study 级聚合 ----------------
def aggregate_by_study(rows, probs, how="mean", labels=None):
    """把图像级预测聚合成 study 级（评分与提交单位都是 Study_id）。

    多张图取均值属于「患者平滑」，规则补充七明确：确定性聚合、不拟合参数，
    性质等同 TTA/集成，属允许的推理后处理。
    """
    if how not in ("mean", "max"):
        raise ValueError(f"不支持的聚合方式: {how}（可选 mean/max）")
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


# ---------------- 提交文件 ----------------
def write_submission(study_rows, study_probs, out_path,
                     all_classes=True, threshold=0.5, decimals=6):
    """写 submission.csv 长表：Subject_id, Study_id, Predict_class, Probability。"""
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
                w.writerow([meta["Subject_id"], meta["Study_id"], cid,
                            f"{p:.{decimals}f}"])
                n_rows += 1
    return n_rows


def check_submission(out_path, test_rows, all_classes=True):
    """提交前自检：行数、study 覆盖、类别 id、概率范围。返回错误列表（空=通过）。"""
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
