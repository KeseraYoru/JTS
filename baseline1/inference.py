# -*- coding: utf-8 -*-
"""推理核心逻辑（**predict.py 和 cost.py 共用同一份**，这是规则第 7 条的硬要求）。

规则原文（第 7 条 + 补充九(三)）：
「cost.py 必须导入并调用与 predict.py 相同的推理函数/模型加载逻辑」，并且
「样本级结果（二值判定及概率排序）必须完全一致，不一致张数须为 0」，
「单张概率绝对误差 |Δp|≤1e-3 视为精度/浮点正常波动」。

因此这里把「加载模型」和「单图推理」抽成公共函数，两个脚本都 import 它，
从结构上保证两者走同一条代码路径——而不是各写一遍再祈祷结果一致。

★ 2026-11-02 重要修正：读图必须支持 JPEG draft 降采样解码
------------------------------------------------------------------
事故记录：predict.py 原先读 data600 的 600² 小图，而 cost.py 读官方 2500x3056 原图。
两条重采样链不同（两次 vs 一次），实测 10 张里有 5 张不一致、|Δp| 最高 0.0122，
是规则容差 1e-3 的 12 倍 —— 核验会直接判不一致。

修法：两条路**都读原图**，并统一用 `Image.draft()` 让 libjpeg 按 1/N 尺度解码
（3056px → 764 或 1528）。好处有两层：
1. 一致性：predict 与 cost 走完全相同的像素，|Δp| 恒为 0；
2. 速度：解码量降到 1/16 左右，且在 T4 上尤其关键（原图解码是 100ms 预算的大头）。
   这是纯「优化推理实现」，规则第 7 条明确允许，不动权重、不改预测语义。

确定性保证（规则第 6 条）：
- model.eval()：关闭 dropout，BN/LN 用训练期统计量；
- torch.no_grad()：不做任何梯度更新；
- 变换全部是确定性算子（TTA 的 6 种也是固定的），不含随机增强。
"""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import IMG_SIZE, NUM_CLASSES, PREP_SIZE
from dataset import make_eval_tfms, make_tta_tfms
from models import build_model

# draft 解码的目标尺度：与训练时 data600 对齐（600 比任何输入尺寸都大，够用）
DEFAULT_SRC_SIZE = PREP_SIZE


def load_model_for_inference(ckpt_path, device, backbone=None, img_size=None,
                             num_classes=NUM_CLASSES):
    """加载推理模型：从 ckpt 里读骨干/尺寸元信息，权重直接灌入（不下载预训练）。

    返回 (model, meta)。meta 里额外带回 src_size，供调用方统一传参。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    backbone = backbone or ckpt.get("backbone")
    img_size = img_size or ckpt.get("img_size", IMG_SIZE)
    num_classes = ckpt.get("num_classes", num_classes)
    src_size = ckpt.get("src_size", DEFAULT_SRC_SIZE)
    if not backbone:
        raise ValueError(
            f"{ckpt_path} 里没有 backbone 信息，请用 --backbone 指定（例如 convnext_base）")

    model = build_model(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state)
    model.eval().to(device)
    meta = {"backbone": backbone, "img_size": img_size,
            "num_classes": num_classes, "src_size": src_size,
            "epoch": ckpt.get("epoch"), "study_auc": ckpt.get("study_auc")}
    return model, meta


def _load_one(path, tfm, src_size=None):
    """读一张图并做确定性变换（PIL 只读一次，转 RGB 三通道）。

    src_size 给定时先用 JPEG 的 draft 让 libjpeg 按 1/N 尺度解码：
    反正最终要缩到 img_size，没必要把 900 万像素全解出来。

    ⚠ predict.py 与 cost.py 必须传同一个 src_size，否则两边像素不同、
      一致性核验（规则第 7 条）就过不了。默认值统一走 DEFAULT_SRC_SIZE。
    """
    with Image.open(path) as im:
        if src_size:
            # draft 只对 JPEG 生效；PNG 会静默忽略，不影响正确性。
            # 对已经是 src_size 见方的缓存图（data600）是一次 no-op。
            im.draft("RGB", (src_size, src_size))
        return tfm(im.convert("RGB"))


@torch.no_grad()
def predict_paths(model, paths, img_size, device, tta=True, batch_size=32,
                  use_amp=False, desc=None, src_size=DEFAULT_SRC_SIZE):
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
            batch = torch.stack([_load_one(p, tfm, src_size) for p in chunk]).to(device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(batch)
            outs.append(torch.sigmoid(logits).float().cpu().numpy())
        probs = np.concatenate(outs, axis=0)
        total = probs if total is None else total + probs
    return total / len(tfms)


@torch.no_grad()
def predict_one_path(model, path, img_size, device, tta=True, use_amp=False,
                     src_size=DEFAULT_SRC_SIZE):
    """单张端到端推理（cost.py 计时用）：读图 → 变换 → 前向 → 概率。"""
    return predict_paths(model, [path], img_size, device, tta=tta,
                         batch_size=1, use_amp=use_amp, src_size=src_size)[0]


@torch.no_grad()
def predict_one_path_multi(models, path, img_size, device, tta=False, use_amp=False,
                           src_size=DEFAULT_SRC_SIZE):
    """**多模型共享一次读图与预处理**，返回各模型概率的平均。

    为什么需要它（2026-11-04 实测踩到）：
    cost.py 原来对每张图「每个模型各调一次 predict_one_path」，于是同一张
    2500x3056 的原图被 **draft 解码 + resize 了 N 遍**。解码恰是 100ms 预算的大头，
    双模型实测 64ms 里有约 20ms 是完全重复的劳动。

    共享后：解码 + resize 只做一次，张量喂给所有模型。
    **预测结果不变**（同一份像素、同一组权重，只是少算了几次重复预处理），
    属于规则第 7 条允许的「优化推理实现」，不需要重训、不影响 submission。

    数学上与旧写法等价：旧写法是「先对每个模型做 TTA 平均、再对模型取平均」，
    这里是「先对每个 TTA 视图做模型平均、再对视图取平均」——都是线性均值，
    差异只有浮点求和顺序（~1e-8 量级，远小于容差 1e-3）。
    """
    tfms = make_tta_tfms(img_size) if tta else [make_eval_tfms(img_size)]
    total = None
    for tfm in tfms:
        # ★ 关键：读图与预处理在整个（视图 × 模型）循环里只出现这一次
        x = _load_one(path, tfm, src_size).unsqueeze(0).to(device)
        acc = None
        for m in models:
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = m(x)
            p = torch.sigmoid(logits).float().cpu().numpy()[0]
            acc = p if acc is None else acc + p
        avg = acc / len(models)
        total = avg if total is None else total + avg
    return total / len(tfms)
