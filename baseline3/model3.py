# -*- coding: utf-8 -*-
"""CXRNet：骨干 + 池化 + 分类头（可选拼 ViewPosition 元数据）。

为什么要单独一层"池化"而不是直接 mean
--------------------------------
一些胸片方案用 attention pooling（CSAM 一类）替代全局平均：让模型自己学"看哪块区域"，
对气胸这种局部征象比平均池化更敏感。这里提供 avg / attn 两种，默认 avg（与 baseline1 可比），
attn 作为可选消融。

ViewPosition 融合
----------------
把 4 维 one-hot（PA/AP/Lateral/未知）拼到池化后的特征上再进分类头。
这是最轻量的接入方式（不改骨干、不改数据流），也是同类赛事第二名"投影路由"的简化版。
view_dim=0 时完全等价于 baseline1 的网络。
"""
import torch
import torch.nn as nn

from archs import build_backbone


class AttentionPool(nn.Module):
    """门控注意力池化：对空间位置加权求和。"""

    def __init__(self, feat_dim, hidden=128):
        super().__init__()
        self.gate = nn.Linear(feat_dim, hidden)
        self.value = nn.Linear(feat_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feat_map):
        # feat_map: (B, C, H, W) -> (B, N, C)
        x = feat_map.flatten(2).transpose(1, 2)
        a = torch.tanh(self.gate(x)) * torch.sigmoid(self.value(x))
        a = self.out(a).squeeze(-1)                    # (B, N)
        w = torch.softmax(a, dim=1).unsqueeze(-1)      # (B, N, 1)
        return (x * w).sum(dim=1)


class CXRNet(nn.Module):
    def __init__(self, arch="convnext_base", num_classes=10, img_size=384,
                 pretrained="auto", pool="avg", view_dim=0, drop=0.1,
                 verbose=True):
        super().__init__()
        self.backbone, feat_dim, self.info = build_backbone(
            arch, img_size=img_size, pretrained=pretrained, drop=drop, verbose=verbose)
        self.arch = arch
        self.pool_kind = pool
        self.view_dim = int(view_dim)
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if pool == "attn":
            self.pool = AttentionPool(feat_dim)
        elif pool == "avg":
            self.pool = None
        else:
            raise ValueError(f"未知池化方式: {pool}（avg/attn）")

        in_dim = feat_dim + self.view_dim
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, num_classes),
        )
        # 头用小初始化，避免一开始就把预训练特征冲掉
        nn.init.trunc_normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)

    def forward_features(self, x):
        f = self.backbone.forward_features(x)
        # ViT（vendor_dinov2）已经返回池化后的 (B, C)
        if f.ndim == 2:
            return f
        # 返回 token 序列的实现：(B, N, C) -> 取 CLS
        if f.ndim == 3:
            return f[:, 0]
        if self.pool is not None:
            return self.pool(f)         # (B, C)
        return f.mean(dim=(2, 3))       # (B, C)

    def forward(self, x, view=None):
        f = self.forward_features(x)
        if self.view_dim:
            if view is None:
                view = torch.zeros(f.shape[0], self.view_dim,
                                   device=f.device, dtype=f.dtype)
            f = torch.cat([f, view.to(f.dtype)], dim=1)
        return self.head(f)


def get_inner(module):
    """剥掉 DataParallel/DDP 外壳。"""
    return module.module if hasattr(module, "module") else module


def set_backbone_trainable(model, flag):
    """冻结/解冻骨干。

    ⚠ baseline1 踩过：先包 DataParallel 再调这个函数 → 'DataParallel' has no attribute 'backbone'。
    这里统一先解包，所有调用方都安全。
    """
    inner = get_inner(model)
    for p in inner.backbone.parameters():
        p.requires_grad_(flag)


def trainable_param_groups(model, base_lr, head_lr_mult=10.0, weight_decay=0.05,
                           layer_decay=1.0):
    """构造参数组：骨干/头分开 lr；bias 与 norm/LayerScale 不做 weight decay；
    可选逐层 lr 衰减（ViT 微调常用，Transformer 深层的权重需要更小步长）。

    返回 [(params, lr, wd), ...]，学习率在此刻按 base_lr 计算，
    训练中的 cosine 调度会在此基础上按比例缩放。
    """
    import re
    inner = get_inner(model)

    # 1) 找出骨干的最大层号，用于逐层衰减
    depth_of = {}
    max_depth = 0
    for name, _p in inner.backbone.named_parameters():
        m = re.search(r"(?:blocks|stages|layers|layer)\.(\d+)\.", name)
        d = int(m.group(1)) if m else 0
        depth_of[name] = d
        max_depth = max(max_depth, d)

    no_decay_keys = ("bias", "norm", "gamma", "cls_token", "pos_embed")

    groups = {}
    for name, p in inner.backbone.named_parameters():
        if not p.requires_grad:
            continue
        d = depth_of.get(name, 0)
        scale = layer_decay ** (max_depth - d) if layer_decay < 1.0 else 1.0
        lr = base_lr * scale
        wd = 0.0 if any(k in name for k in no_decay_keys) else weight_decay
        key = (round(lr, 10), wd)
        groups.setdefault(key, []).append(p)

    for name, p in inner.head.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if ("bias" in name) else weight_decay
        key = (round(base_lr * head_lr_mult, 10), wd)
        groups.setdefault(key, []).append(p)

    out = []
    for (lr, wd), params in groups.items():
        out.append({"params": params, "lr": lr, "weight_decay": wd})
    return out
