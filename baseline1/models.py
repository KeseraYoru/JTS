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
# convnext_large（197M 参数 / fp16 约 394MB）是唯一「单模型就能压进 500MB 且算力翻倍」的选项：
# 双模型 @448 在 T4 上推算 111ms 有超限风险，而 large 单模型 @384 约 73ms，同时精度预期更高。
BACKBONE_CHOICES = ("convnext_base", "convnext_small", "convnext_large",
                    "convnext_tiny", "efficientnet_v2_m", "efficientnet_v2_s",
                    "swin_b", "swin_t", "vit_b_16", "resnet50")


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
    """冻结/解冻骨干（只影响 requires_grad，不改变已学到的权重）。

    ⚠ 必须先用 get_model() 剥掉 DataParallel：train.py 在多卡下把模型包成
      DataParallel，而 DataParallel 对象本身没有 .backbone 属性，
      直接 model.backbone 会 AttributeError（2026-11-02 用 --freeze-epochs 1 踩到过）。
      注意这不影响 nn.Module 的行为语义，只是属性转发问题。
    """
    inner = get_model(model)
    for p in inner.backbone.parameters():
        p.requires_grad_(flag)


# ---------------- 加载自监督预训练权重 ----------------
def load_ssl_backbone(model, ssl_path):
    """把自监督阶段训练好的骨干权重灌进模型（规则补充四允许）。

    ⚠ 必须按「键名 + 形状」双重过滤后再灌：
      `load_state_dict(strict=False)` 只容忍「某边多键/缺键」，
      但**同名而形状不同会直接抛 RuntimeError**。
      典型场景：拿 convnext 的 SSL 权重去初始化 efficientnet ——
      第一层卷积 convnext 是 4x4/128ch，efficientnet_v2_m 是 3x3/24ch，
      于是 features.0.0.weight 形状不匹配 → 整个训练启动失败（2026-11-03 踩到）。
    """
    ckpt = torch.load(ssl_path, map_location="cpu", weights_only=False)
    state = ckpt.get("backbone", ckpt)
    # 兼容保存时带 "backbone." 前缀的情况
    if state and next(iter(state)).startswith("backbone."):
        state = {k[len("backbone."):]: v for k, v in state.items()}

    target = model.backbone.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in target and tuple(target[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    model.backbone.load_state_dict(filtered, strict=False)
    print(f"[SSL] {ssl_path}")
    print(f"[SSL] 已加载 {len(filtered)} 个键 | 跳过 {len(skipped)} 个（形状或命名不匹配）")
    if not filtered:
        print("[SSL] ⚠ 没有任何键匹配 —— 该骨干与 SSL 权重不是同一架构，"
              "本次等同纯 ImageNet 起点（这不算错误，但要在台账里注明）")
    elif len(filtered) < 0.5 * len(target):
        print(f"[SSL] ⚠ 只匹配到 {len(filtered)}/{len(target)} 个键，"
              "迁移效果会打折扣")
    return model


# ---------------- 体积信息（合规第 8 条自查） ----------------
def param_stats(model):
    """返回 (总参数量, fp32 权重体积 MB)。"""
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / 1024 / 1024


def get_model(module):
    """剥掉 DataParallel 包装。"""
    return module.module if hasattr(module, "module") else module
