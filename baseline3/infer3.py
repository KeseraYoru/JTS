# -*- coding: utf-8 -*-
"""推理核心（predict3.py 与 cost3.py **共用同一份**，规则第 7 条硬要求）。

三条从 baseline1 继承、不能再犯的约束
----------------------------------
1. **读原图 + draft 解码**：官方核验读的是 2500×3056 原图；用 data600 小图测出来的
   耗时是假数字（我们曾因此得出 16ms 的错误结论，骗了两周）。
2. **predict 与 cost 必须走同一串像素**：任何一侧多一次 resize，概率就会差 0.01 量级，
   直接越过 1e-3 的容差（实测 10 张里 5 张不一致）。
3. **多模型共享一次预处理**：同一张原图被解码 N 遍纯属浪费；共享后双模型耗时
   从 64ms 降到 42ms，且预测结果逐位等价。

新增：ViewPosition one-hot 参与前向（模型训练时若用了它，推理必须一起喂）。
"""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common3 import encode_view
from config3 import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from data3 import make_eval_tfms, make_tta_tfms
from model3 import CXRNet


def load_from_ckpt(ckpt_path, device, use_ema=True, img_size_override=None):
    """从 ckpt 的元信息重建结构并灌权重。

    `impl / arch / img_size / view_dim / pool` 全部来自 ckpt —— 不再靠命令行猜，
    这是 baseline1 里「加了融合层就 unexpected keys」那类问题的根治办法。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    if not arch:
        raise ValueError(f"{ckpt_path} 里没有 arch 字段（是不是 baseline1 的旧权重？）")
    img_size = int(img_size_override or ckpt.get("img_size", 384))
    view_dim = int(ckpt.get("view_dim", 0))
    pool = ckpt.get("pool", "avg")

    model = CXRNet(arch=arch, num_classes=ckpt.get("num_classes", NUM_CLASSES),
                   img_size=img_size, pretrained="none", pool=pool,
                   view_dim=view_dim, drop=0.0, verbose=False)
    state = ckpt.get("ema_state_dict") if use_ema else ckpt.get("model_state_dict")
    state = state or ckpt.get("model_state_dict")
    target = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in target and tuple(target[k].shape) == tuple(v.shape)}
    model.load_state_dict(filtered, strict=False)
    model.eval().to(device)

    meta = {
        "arch": arch, "img_size": img_size, "view_dim": view_dim, "pool": pool,
        "src_size": int(ckpt.get("src_size", 600)),
        "epoch": ckpt.get("epoch"), "study_auc": ckpt.get("study_auc"),
        "impl": ckpt.get("impl"), "ckpt": ckpt_path,
        "loaded_keys": len(filtered), "total_keys": len(target),
    }
    return model, meta


def load_image(path, tfm, src_size=None):
    """读一张图 + 确定性变换。draft 让 libjpeg 按 1/2、1/4、1/8 档解码。

    注意 draft 只有这三档：要 448 和要 600 都会落到 1/4（625px），
    所以「把 draft 目标改成 img_size 来省时间」是不成立的。
    """
    with Image.open(path) as im:
        if src_size:
            im.draft("RGB", (src_size, src_size))
        return tfm(im.convert("RGB"))


def _view_tensor(rows, idx):
    v = encode_view([rows[idx]])
    return torch.from_numpy(v).float()


@torch.no_grad()
def predict_paths(model, paths, rows, img_size, device, tta_kinds=("base",),
                  batch_size=16, use_amp=True, desc=None, src_size=600):
    """对一批路径推理，返回 [N, 10] 概率（已做 TTA 平均）。"""
    tfms = make_tta_tfms(img_size, tta_kinds)
    total = None
    for ti, tfm in enumerate(tfms):
        outs = []
        it = range(0, len(paths), batch_size)
        if desc:
            it = tqdm(it, desc=f"{desc} TTA{ti + 1}/{len(tfms)}" if len(tfms) > 1 else desc)
        for i in it:
            chunk = paths[i:i + batch_size]
            batch = torch.stack([load_image(p, tfm, src_size) for p in chunk]).to(device)
            view = (torch.cat([_view_tensor(rows, i + j) for j in range(len(chunk))])
                    .to(device) if model.view_dim else None)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(batch, view)
            outs.append(torch.sigmoid(logits).float().cpu().numpy())
        probs = np.concatenate(outs, axis=0)
        total = probs if total is None else total + probs
    return total / len(tfms)


def fuse_probs(plist, method="mean"):
    """逐样本融合多模型概率：mean（算术平均）| logit（log-odds 平均）。

    两者都是**逐样本**运算，满足规则第 6 条；秩平均需要跨样本排秩，不可用于提交。
    """
    if len(plist) == 1:
        return plist[0]
    if method == "logit":
        z = np.mean([np.log(np.clip(p, 1e-6, 1 - 1e-6))
                     - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6)) for p in plist], axis=0)
        return 1.0 / (1.0 + np.exp(-z))
    return np.mean(plist, axis=0)


@torch.no_grad()
def predict_one_multi(models, path, row, img_size, device, use_amp=True,
                      src_size=600, method="mean"):
    """单张 + 多模型：**读图与预处理只做一次**，张量喂给所有模型。

    与「每个模型各读一遍」数学等价（都是对同一份像素做前向再平均），
    但省掉了 N-1 次 JPEG 解码 —— 双模型实测 64ms → 42ms。
    """
    tfm = make_eval_tfms(img_size)
    x = load_image(path, tfm, src_size).unsqueeze(0).to(device)
    view = None
    if any(m.view_dim for m in models):
        view = torch.from_numpy(encode_view([row])).float().to(device)
    plist = []
    for m in models:
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = m(x, view if m.view_dim else None)
        plist.append(torch.sigmoid(logits).float().cpu().numpy()[0])
    return fuse_probs(plist, method)


MEAN = IMAGENET_MEAN
STD = IMAGENET_STD
