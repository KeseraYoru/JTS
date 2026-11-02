# -*- coding: utf-8 -*-
"""推理核心逻辑（**predict.py 和 cost.py 共用同一份**，这是规则第 7 条的硬要求）。

规则原文（第 7 条 + 补充九(三)）：
「cost.py 必须导入并调用与 predict.py 相同的推理函数/模型加载逻辑」，并且
「样本级结果（二值判定及概率排序）必须完全一致，不一致张数须为 0」，
「单张概率绝对误差 |Δp|≤1e-3 视为精度/浮点正常波动」。

因此这里把「加载模型」和「单图推理」抽成公共函数，两个脚本都 import 它，
从结构上保证两者走同一条代码路径——而不是各写一遍再祈祷结果一致。

确定性保证（规则第 6 条）：
- model.eval()：关闭 dropout，BN/LN 用训练期统计量；
- torch.no_grad()：不做任何梯度更新；
- 变换全部是确定性算子（TTA 的 6 种也是固定的），不含随机增强。
"""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import IMG_SIZE, NUM_CLASSES
from dataset import make_eval_tfms, make_tta_tfms
from models import build_model


def load_model_for_inference(ckpt_path, device, backbone=None, img_size=None,
                             num_classes=NUM_CLASSES):
    """加载推理模型：从 ckpt 里读骨干/尺寸元信息，权重直接灌入（不下载预训练）。

    返回 (model, meta)。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    backbone = backbone or ckpt.get("backbone")
    img_size = img_size or ckpt.get("img_size", IMG_SIZE)
    num_classes = ckpt.get("num_classes", num_classes)
    if not backbone:
        raise ValueError(
            f"{ckpt_path} 里没有 backbone 信息，请用 --backbone 指定（例如 convnext_base）")

    model = build_model(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state)
    model.eval().to(device)
    meta = {"backbone": backbone, "img_size": img_size,
            "num_classes": num_classes,
            "epoch": ckpt.get("epoch"), "study_auc": ckpt.get("study_auc")}
    return model, meta


def _load_one(path, tfm):
    """读一张图并做确定性变换（PIL 只读一次，转 RGB 三通道）。"""
    with Image.open(path) as im:
        return tfm(im.convert("RGB"))


@torch.no_grad()
def predict_paths(model, paths, img_size, device, tta=True, batch_size=32,
                  use_amp=False, desc=None):
    """对一组**绝对路径**做推理，返回 [N, 10] 概率。

    tta=True 时按 6 种确定性变换分别推理后取平均（规则允许）。
    batch_size=1 + 无进度条 = cost.py 的串行计时用法。
    """
    tfms = make_tta_tfms(img_size) if tta else [make_eval_tfms(img_size)]
    total = None
    for ti, tfm in enumerate(tfms):
        outs = []
        iterator = range(0, len(paths), batch_size)
        if desc:
            # ⚠ 这里必须用外层的 tfm 序号 ti，不能用内层 batch 序号 i
            #   （2026-10-31 踩过：写成 i 会 UnboundLocalError，且只在传 desc 时才触发，
            #    所以 cost.py 的 desc=None 路径测不出来，predict.py 一跑就崩）
            iterator = tqdm(iterator,
                            desc=f"{desc} TTA{ti + 1}/{len(tfms)}" if tta else desc)
        for i in iterator:
            chunk = paths[i:i + batch_size]
            batch = torch.stack([_load_one(p, tfm) for p in chunk]).to(device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(batch)
            outs.append(torch.sigmoid(logits).float().cpu().numpy())
        probs = np.concatenate(outs, axis=0)
        total = probs if total is None else total + probs
    return total / len(tfms)


@torch.no_grad()
def predict_one_path(model, path, img_size, device, tta=True, use_amp=False):
    """单张端到端推理（cost.py 计时用）：读图 → 变换 → 前向 → 概率。"""
    return predict_paths(model, [path], img_size, device, tta=tta,
                         batch_size=1, use_amp=use_amp)[0]
