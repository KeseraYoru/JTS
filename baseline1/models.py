# -*- coding: utf-8 -*-
"""模型：torchvision ImageNet 骨干 + 10 类多标签头。

合规要点（规则第 1 条）：
- 只用 torchvision 自带的 **ImageNet-1k** 预训练权重（通用自然图像，非医学）；
- 不使用 EVA-X / CheXFound / RETFound 等任何以医学影像为主数据集预训练的权重；
- 骨干全部是 torchvision 内置实现，复赛容器（只保证 torch+torchvision）可直接加载。

结构：backbone（去掉原 1000 类头）+ Dropout + Linear(10)。
这样拆开的好处是参数分组清晰，可以给骨干和头设不同学习率。
"""
import torch
import torch.nn as nn

from config import BACKBONE, DROPOUT, HEAD_LR_MULT, LR, NUM_CLASSES, WEIGHT_DECAY

# 允许的骨干（都来自 torchvision，ImageNet-1k 预训练）
BACKBONE_CHOICES = ("convnext_base", "convnext_small", "efficientnet_v2_m",
                    "swin_b", "vit_b_16", "resnet50")


def _build_vision(name, pretrained=True):
    """构建 torchvision 模型（pretrained=True 时下载 ImageNet-1k 权重）。"""
    from torchvision import models as tvm
    # 兼容不同 torchvision 版本：get_model 才是统一入口
    weights = "DEFAULT" if pretrained else None
    try:
        return tvm.get_model(name, weights=weights)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"构建骨干 {name} 失败：{e}\n"
            f"可选：{BACKBONE_CHOICES}\n"
            "提示：首次使用需要下载 ImageNet 权重（约 80~350MB），"
            "离线环境可先在有网机器上下载后放到 torch hub 缓存目录。"
        ) from e


def _strip_head(model, name):
    """把原 1000 类分类头换成 Identity，返回特征维度。"""
    if name.startswith("convnext"):
        dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
    elif name.startswith("efficientnet"):
        dim = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()
    elif name.startswith("swin"):
        dim = model.head.in_features
        model.head = nn.Identity()
    elif name.startswith("vit"):
        dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
    elif name.startswith("resnet"):
        dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError(f"未支持的骨干: {name}（可选 {BACKBONE_CHOICES}）")
    return dim


class ChestXrayModel(nn.Module):
    """backbone + dropout + Linear(10)。forward 输出 10 个 logits。"""

    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                 pretrained=True, dropout=DROPOUT):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.backbone = _build_vision(backbone_name, pretrained)
        feat_dim = _strip_head(self.backbone, backbone_name)
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(backbone_name=BACKBONE, num_classes=NUM_CLASSES,
                pretrained=True):
    """供训练/推理统一调用的工厂函数。"""
    if backbone_name not in BACKBONE_CHOICES:
        raise ValueError(f"不支持的骨干 {backbone_name}（可选 {BACKBONE_CHOICES}）")
    return ChestXrayModel(backbone_name=backbone_name, num_classes=num_classes,
                          pretrained=pretrained)


# ---------------- 参数组 / 冻结 ----------------
def _split_decay(named_params):
    """把参数分成「需要 weight decay」和「不需要」两组（bias / norm 不衰减）。"""
    decay, no_decay = [], []
    for _, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return decay, no_decay


def build_optimizer(model, lr=LR, head_lr_mult=HEAD_LR_MULT,
                    weight_decay=WEIGHT_DECAY):
    """AdamW，骨干与头分不同学习率；bias/norm 不做 weight decay。"""
    b_decay, b_norm = _split_decay(model.backbone.named_parameters())
    h_decay, h_norm = _split_decay(model.head.named_parameters())
    groups = []
    if b_decay:
        groups.append({"params": b_decay, "lr": lr,
                       "weight_decay": weight_decay})
    if b_norm:
        groups.append({"params": b_norm, "lr": lr, "weight_decay": 0.0})
    if h_decay:
        groups.append({"params": h_decay, "lr": lr * head_lr_mult,
                       "weight_decay": weight_decay})
    if h_norm:
        groups.append({"params": h_norm, "lr": lr * head_lr_mult,
                       "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def set_backbone_trainable(model, flag):
    """冻结/解冻骨干（只影响 requires_grad，不改变已学到的权重）。"""
    for p in model.backbone.parameters():
        p.requires_grad_(flag)


# ---------------- 加载自监督预训练权重 ----------------
def load_ssl_backbone(model, ssl_path):
    """把自监督阶段训练好的骨干权重灌进模型（规则补充四允许）。"""
    ckpt = torch.load(ssl_path, map_location="cpu", weights_only=False)
    state = ckpt.get("backbone", ckpt)
    # 兼容保存时带 "backbone." 前缀的情况
    if state and next(iter(state)).startswith("backbone."):
        state = {k[len("backbone."):]: v for k, v in state.items()}
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"[SSL] 已加载自监督骨干: {ssl_path} | 缺失 {len(missing)} | "
          f"多余 {len(unexpected)}")
    return model


# ---------------- 体积信息（合规第 8 条自查） ----------------
def param_stats(model):
    """返回 (总参数量, fp32 权重体积 MB)。"""
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / 1024 / 1024


def get_model(module):
    """剥掉 DataParallel 包装。"""
    return module.module if hasattr(module, "module") else module
