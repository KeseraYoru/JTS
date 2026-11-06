# -*- coding: utf-8 -*-
"""骨干注册表：统一对外提供 forward_features() + num_features。

三条来源
-------
1. **torchvision**（ImageNet-1k，BSD-3）：convnext_* / efficientnet_v2_* / swin_*
   —— 容器原生支持，最稳的一档，作为基线。
2. **vendor ConvNeXt**：加载 ImageNet-22k 权重（timm/官方 release 命名）。
3. **vendor DINOv2 ViT**：网页级自监督权重（LVD-142M，Apache-2.0）。

统一的特征接口约定
----------------
- 卷积类骨干返回 (B, C, H, W)；
- ViT 类骨干返回 (B, C)；
- model3 依据 ndim 决定池化方式，因此新增骨干不用改 model。

权重来源全部登记在 config3.PRETRAIN_SOURCES，申报 A1~A6 时直接取用。
"""
import os

import torch
import torch.nn as nn

from config3 import PRETRAIN_SOURCES, WEIGHTS_DIR
from vendor.convnext_timm import ConvNeXtTimm, load_convnext_weights
from vendor.vit_dino import DINOv2ViT, load_dinov2_weights

# torchvision 白名单：都有 .features(x) 且返回卷积特征图
TV_ARCHS = (
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "efficientnet_v2_s", "efficientnet_v2_m",
    "swin_t", "swin_b",
)
VENDOR_ARCHS = ("convnext_base_in22k", "dinov2_vits14", "dinov2_vitb14")

ALL_ARCHS = TV_ARCHS + VENDOR_ARCHS


class TVBackbone(nn.Module):
    """torchvision 骨干包装：只暴露特征图，分类头留空不用。"""

    def __init__(self, name, pretrained):
        super().__init__()
        from torchvision import models as tvm
        weights = "DEFAULT" if pretrained == "auto" else None
        self.model = tvm.get_model(name, weights=weights)
        self.name = name
        if name.startswith("convnext"):
            self.num_features = self.model.classifier[2].in_features
        elif name.startswith("efficientnet"):
            self.num_features = self.model.classifier[1].in_features
        elif name.startswith("swin"):
            self.num_features = self.model.head.in_features
        else:
            raise ValueError(f"未支持的 torchvision 骨干: {name}")

    def forward_features(self, x):
        f = self.model.features(x)
        if f.ndim == 4 and f.shape[1] != self.num_features and f.shape[-1] == self.num_features:
            f = f.permute(0, 3, 1, 2)      # Swin: (B,H,W,C) -> (B,C,H,W)
        return f


def _download(url, filename):
    """下载到 weights/，已存在则跳过。用 torch.hub 的工具，不引入 requests。"""
    dst = os.path.join(WEIGHTS_DIR, filename)
    if os.path.exists(dst):
        print(f"[权重] 已存在，跳过下载: {dst}")
        return dst
    print(f"[权重] 下载中: {url}")
    torch.hub.download_url_to_file(url, dst, progress=True)
    return dst


def resolve_pretrained_path(arch, pretrained):
    """把 --pretrained 解析成具体权重文件路径（或 None 表示随机初始化）。"""
    if pretrained in (None, "", "none"):
        return None
    if os.path.exists(str(pretrained)):
        return str(pretrained)
    if pretrained == "auto":
        src = PRETRAIN_SOURCES.get(arch)
        if not src:
            return None       # torchvision 会自己下载，不需要我们管
        return _download(src["url"], os.path.basename(src["url"]))

    # 走到这里说明：不是 auto/none 之外的关键字，而且这个路径不存在。
    # 说得具体一点，否则会误导成"参数写错了"（我们刚踩过）。
    if "/" in str(pretrained) or str(pretrained).endswith((".pth", ".pt", ".bin")):
        raise FileNotFoundError(
            f"权重文件不存在: {pretrained}\n"
            "排查顺序：\n"
            "  ① 是否跑过 `bash download_weights.sh`？\n"
            "  ② 服务器若报 `certificate has expired`（本机 CA 过期/代理拦截），"
            "脚本会自动用 curl -k 重试；也可手动：\n"
            "     curl -k -L -o weights/dinov2_vitb14_pretrain.pth \\\n"
            "       https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth\n"
            "  ③ 或在本机下载后 scp 到 weights/（路径要对上 --weights 传的那个）")
    raise ValueError(f"--pretrained 只能是 auto / none / 本地路径，收到 {pretrained}")


def build_backbone(arch, img_size=384, pretrained="auto", drop=0.0, verbose=True):
    """返回 (backbone, feat_dim, info)。

    info 里带 impl 字段，写进 ckpt，推理端据此重建完全相同的结构。
    """
    if arch in TV_ARCHS:
        if pretrained not in ("auto", "none", None) and os.path.exists(str(pretrained)):
            print(f"⚠ torchvision 骨干（{arch}）目前只支持 --pretrained auto/none，"
                  "忽略传入的本地路径；如需从本地权重起步，请用 vendor 架构"
                  "（convnext_base_in22k / dinov2_vitb14）")
        bb = TVBackbone(arch, pretrained)
        src = PRETRAIN_SOURCES.get(arch, {})
        info = {"impl": "torchvision", "arch": arch, "img_size": img_size,
                "license": src.get("license", "BSD-3-Clause"),
                "upstream": src.get("upstream", "ImageNet-1k（通用自然图像）"),
                "url": src.get("url", ""),
                "loaded_from": "torchvision hub cache" if pretrained == "auto" else "random"}
        return bb, bb.num_features, info

    if arch == "convnext_base_in22k":
        bb = ConvNeXtTimm("convnext_base")
        path = resolve_pretrained_path(arch, pretrained)
        if path:
            load_convnext_weights(bb, path, verbose=verbose)
        src = PRETRAIN_SOURCES[arch]
        info = {"impl": "vendor_convnext", "arch": "convnext_base", "img_size": img_size,
                "license": src["license"], "upstream": src["upstream"],
                "url": src["url"], "loaded_from": path or "random"}
        return bb, bb.num_features, info

    if arch in ("dinov2_vits14", "dinov2_vitb14"):
        bb = DINOv2ViT(arch, img_size=img_size, drop=drop)
        path = resolve_pretrained_path(arch, pretrained)
        if path:
            load_dinov2_weights(bb, path, verbose=verbose)
        src = PRETRAIN_SOURCES[arch]
        info = {"impl": "vendor_dinov2", "arch": arch, "img_size": img_size,
                "license": src["license"], "upstream": src["upstream"],
                "url": src["url"], "loaded_from": path or "random"}
        return bb, bb.num_features, info

    raise ValueError(f"未知骨干 {arch}，可选：{ALL_ARCHS}")


def load_ssl_into_backbone(backbone, ssl_path, verbose=True):
    """把自监督阶段训好的骨干权重灌进来（官方训练集自监督，规则补充四允许）。

    与 baseline1 的差异：这里先按「键名 + 形状」双重过滤。
    strict=False 只容忍键多键少，同名不同形状会直接抛 RuntimeError，
    换架构时必踩（我们已经踩过一次）。
    """
    if not ssl_path:
        return 0
    ckpt = torch.load(ssl_path, map_location="cpu", weights_only=False)
    state = ckpt.get("backbone", ckpt.get("state_dict", ckpt))
    if any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items()}

    target = backbone.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in target and tuple(target[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)
    backbone.load_state_dict(filtered, strict=False)

    if verbose:
        print(f"[SSL] {ssl_path} | 匹配 {len(filtered)}/{len(target)} 键 | 跳过 {len(skipped)}")
        if not filtered:
            print("[SSL] ⚠ 一个键都没匹配上 —— 该 SSL 权重与本骨干不同架构，"
                  "本次等同纯预训练起点（不是错误，但要在台账里注明）")
    return len(filtered)


def param_stats(model):
    """参数量与 fp16 体积，用于合规自查（≤500MB）。"""
    n = sum(p.numel() for p in model.parameters())
    return {"params_m": n / 1e6, "fp16_mb": n * 2 / 1024 / 1024,
            "fp32_mb": n * 4 / 1024 / 1024}
