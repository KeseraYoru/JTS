# -*- coding: utf-8 -*-
"""DINOv2 ViT-B/14（纯 torch 实现），用于加载网页级自监督权重。

为什么值得试
-----------
DINOv2 的预训练数据是 LVD-142M（从公开网页图像自动筛选构建），**不是医学数据**，
主数据集未被列为医学 → 按规则第 1 条 + 补充二(一) 属于合规权重；
许可证为 Apache-2.0（GitHub 官方 release 版本），非 NC、非 gated。
而规则原文点名 "LAION 等通用自然图像" 合规，DINOv2 正是这一路线的代表。

结构要点（与官方 state_dict 严格对应，便于直接 load）
--------------------------------------------------
    patch_embed.proj      Conv2d(3, dim, 14, 14)
    cls_token / pos_embed [1,1,dim] / [1,1+N,dim]（官方为 518px，需插值）
    blocks.{i}.norm1 / attn.qkv / attn.proj / ls1.gamma / norm2 / mlp.fc1 / mlp.fc2 / ls2.gamma
    norm                  LayerNorm(dim, eps=1e-6)（DINOv2 用 1e-6，不是默认 1e-5）

注意：官方权重里还有一个 `mask_token`（MAE 式蒸馏用），我们直接丢弃。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=384, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, C, gh, gw)
        return x.flatten(2).transpose(1, 2)   # (B, N, C)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LayerScale(nn.Module):
    def __init__(self, dim, init=1.0):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init=1.0)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init=1.0)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


_DINO_CFG = {
    "dinov2_vits14": dict(embed_dim=384, depth=12, heads=6, patch=14),
    "dinov2_vitb14": dict(embed_dim=768, depth=12, heads=12, patch=14),
}


class DINOv2ViT(nn.Module):
    """输出 CLS token 特征 (B, C)；分类头由 model3 接。"""

    def __init__(self, variant="dinov2_vitb14", img_size=384, drop=0.0):
        super().__init__()
        if variant not in _DINO_CFG:
            raise ValueError(f"未支持的 DINOv2 变体: {variant}")
        cfg = _DINO_CFG[variant]
        self.variant = variant
        self.embed_dim = cfg["embed_dim"]
        self.num_features = cfg["embed_dim"]
        self.patch_embed = PatchEmbed(img_size, cfg["patch"], 3, cfg["embed_dim"])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg["embed_dim"]))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, cfg["embed_dim"]))
        # 随机初始化时也要给个合理起点（否则 pos_embed 全零，等价于没有位置信息）
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            Block(cfg["embed_dim"], cfg["heads"], drop=drop) for _ in range(cfg["depth"])])
        self.norm = nn.LayerNorm(cfg["embed_dim"], eps=1e-6)

    def interpolate_pos_encoding(self, x, h, w):
        """官方 pos_embed 是 518px（37x37）；我们的输入不同，需双三次插值。"""
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        gh = gw = int(math.sqrt(n))
        patch_pos = patch_pos.reshape(1, gh, gw, dim).permute(0, 3, 1, 2)
        new_gh, new_gw = h, w
        patch_pos = F.interpolate(patch_pos, size=(new_gh, new_gw),
                                  mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_gh * new_gw, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)                       # (B, N, C)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)                # (B, 1+N, C)
        gh = gw = int(math.sqrt(x.shape[1] - 1))
        x = x + self.interpolate_pos_encoding(x, gh, gw)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                # CLS token

    def forward(self, x):
        return self.forward_features(x)


def _interp_pos_embed(pe, new_patches):
    """把官方 518px 的 pos_embed（1, 1+1369, C）插值到我们的 patch 数。

    ★ 必须真的插值并写回，不能"跳过 pos_embed"：跳过的话模型保留随机初始化，
    预训练权重等于只加载了一半，而且指标不会报错，只会莫名其妙地低。
    """
    if pe.shape[1] - 1 == new_patches:
        return pe
    cls_pe = pe[:, :1]
    patch_pe = pe[:, 1:]
    dim = pe.shape[-1]
    n = patch_pe.shape[1]
    gh = gw = int(round(math.sqrt(n)))
    patch_pe = patch_pe.reshape(1, gh, gw, dim).permute(0, 3, 1, 2).float()
    new_hw = int(round(math.sqrt(new_patches)))
    patch_pe = F.interpolate(patch_pe, size=(new_hw, new_hw),
                             mode="bicubic", align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_patches, dim)
    return torch.cat([cls_pe.float(), patch_pe], dim=1)


def load_dinov2_weights(model, state_path, verbose=True):
    """加载官方 DINOv2 权重：丢 mask_token、插值 pos_embed、按形状过滤。"""
    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k: v for k, v in state.items() if not k.startswith("mask_token")}
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    if any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items()}

    pos_pe = state.pop("pos_embed", None)
    target = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in target and tuple(target[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if pos_pe is not None:
        with torch.no_grad():
            model.pos_embed.copy_(
                _interp_pos_embed(pos_pe, model.pos_embed.shape[1] - 1)
                .to(model.pos_embed.dtype))
        if verbose:
            print(f"[权重] pos_embed 已插值: {tuple(pos_pe.shape)} -> "
                  f"{tuple(model.pos_embed.shape)}")

    if verbose:
        n_t = len(target)
        print(f"[权重] {state_path}")
        print(f"[权重] 匹配 {len(filtered)}/{n_t} 键 | 跳过 {len(skipped)} | "
              f"缺失 {len(missing)} | 多余 {len(unexpected)}")
        if skipped[:5]:
            print(f"[权重] 跳过的键示例: {skipped[:5]}")
        if len(filtered) < 0.6 * n_t:
            print("[权重] ⚠ 匹配率 <60%：可能加载的不是 ViT-B/14，或权重被别的框架包过一层")
    return len(filtered), len(missing), len(skipped)
