# -*- coding: utf-8 -*-
"""ConvNeXt（timm 命名版），纯 torch 实现，用于加载 ImageNet-22k 预训练权重。

为什么不用 torchvision 的 convnext
--------------------------------
torchvision 只发布 ImageNet-1k 权重；ImageNet-22k 权重由 timm / 官方 release 提供，
其 state_dict 使用 timm 命名（stem / stages / blocks / norm / head）。
本文件按 timm 的结构与命名实现，因此可以**直接** load_state_dict，
打包时也不依赖 timm。

参数命名对照（官方 release <-> timm <-> 本文件）
-----------------------------------------------
    downsample_layers.0.0.weight   ->  stem.0.weight
    downsample_layers.0.1.weight   ->  stem.1.weight
    downsample_layers.1.0.weight   ->  stages.0.downsample.0.weight
    stages.0.0.dwconv.weight       ->  stages.0.blocks.0.dwconv.weight
加载时由 `remap_convnext_state()` 统一处理这两种历史命名。
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelsLastLN(nn.Module):
    """LayerNorm over channels，输入是 (B, C, H, W)。timm 用 data_format="channels_last"。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ChannelsLastLN(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init else None

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)          # NHWC 给 Linear
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x * self.gamma[None, :, None, None]
        return inp + x


class ConvNeXtStage(nn.Module):
    def __init__(self, dim, out_dim, depth, downsample=True):
        super().__init__()
        self.downsample = nn.Sequential(
            ChannelsLastLN(dim), nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        ) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_dim) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


# (dims, depths)：base / large / tiny 的官方配置
_CFG = {
    "convnext_tiny":  ([96, 192, 384, 768], [3, 3, 9, 3]),
    "convnext_small": ([96, 192, 384, 768], [3, 3, 27, 3]),
    "convnext_base":  ([128, 256, 512, 1024], [3, 3, 27, 3]),
    "convnext_large": ([192, 384, 768, 1536], [3, 3, 27, 3]),
}


class ConvNeXtTimm(nn.Module):
    """输出 (B, C, H, W) 的特征图，分类头由 model3 负责接。"""

    def __init__(self, variant="convnext_base"):
        super().__init__()
        if variant not in _CFG:
            raise ValueError(f"未支持的 ConvNeXt 变体: {variant}")
        dims, depths = _CFG[variant]
        self.variant = variant
        self.num_features = dims[-1]

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            ChannelsLastLN(dims[0]),
        )
        stages = []
        for i in range(4):
            stages.append(ConvNeXtStage(dims[i], dims[i], depths[i], downsample=False)
                          if i == 0 else
                          ConvNeXtStage(dims[i - 1], dims[i], depths[i], downsample=True))
        self.stages = nn.Sequential(*stages)
        self.norm = ChannelsLastLN(dims[-1])

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.norm(x)

    def forward(self, x):
        return self.forward_features(x)


def remap_convnext_state(state):
    """把官方 release / timm 的两种历史命名统一成本文件的命名。

    处理三类差异：
    1. 去掉 "model." / "module." 前缀；
    2. downsample_layers.{i}.{0,1}.*  ->  stem.{0,1}.*（i=0）或 stages.{i-1}.downsample.{0,1}.*
    3. stages.{i}.{j}.* （j 是数字）  ->  stages.{i}.blocks.{j}.*
    最后丢掉 head.*（1000 类分类头，我们不用）。
    """
    out = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("head.") or k.startswith("norm_pre."):
            continue

        parts = k.split(".")
        if parts[0] == "downsample_layers" and len(parts) >= 3:
            i = int(parts[1])
            rest = ".".join(parts[2:])
            if i == 0:
                k = f"stem.{rest}"
            else:
                # ★ 这里必须是 stages.{i}，不是 stages.{i-1}：
                # 官方的 downsample_layers[1] 负责 128->256，落在本实现的 stages[1].downsample。
                # 写成 i-1 会让 12 个下采样权重组静默加载失败（不报错、只是指标偏低）。
                k = f"stages.{i}.downsample.{rest}"
        elif parts[0] == "stages" and len(parts) >= 3 and parts[2].isdigit():
            # stages.0.0.dwconv.weight -> stages.0.blocks.0.dwconv.weight
            k = f"stages.{parts[1]}.blocks.{parts[2]}." + ".".join(parts[3:])
        out[k] = v
    return out


def load_convnext_weights(model, state_path, verbose=True):
    """加载权重，逐键检查形状；只灌入名字与形状都匹配的键。

    baseline1 踩过的坑：strict=False 只容忍"键多键少"，不容忍同名不同形状，
    会直接抛 RuntimeError 把训练打断。这里先过滤再加载，并且把跳过的键打印出来。
    """
    # 体积闸门：断网/代理截断时，curl 可能只拿到几百字节却报"100% 完成"（我们踩过：
    # 264 字节的"权重"，恰好是真文件的前 264 字节，肉眼看不出来）。这里先挡住。
    size_mb = os.path.getsize(state_path) / 1024 / 1024
    if size_mb < 100:
        raise SystemExit(
            f"权重文件只有 {size_mb:.2f} MB，明显不完整（正常应约 340~440MB）：{state_path}\n"
            "常见原因：下载被代理/网络截断，curl 却显示 100%。重下即可：\n"
            "  rm -f 该文件\n"
            "  env -u HTTP_PROXY -u HTTPS_PROXY curl -k -L --http1.1 --retry 5 -C - "
            "-o 该文件 <URL>\n"
            "下载后务必 `ls -lh` 确认体积。")

    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = remap_convnext_state(state)

    target = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in target and tuple(target[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if verbose:
        n_t = len(target)
        print(f"[权重] {state_path}")
        print(f"[权重] 匹配 {len(filtered)}/{n_t} 键 | 跳过 {len(skipped)} | "
              f"缺失 {len(missing)} | 多余 {len(unexpected)}")
        if skipped[:5]:
            print(f"[权重] 跳过的键示例: {skipped[:5]}")
        if len(filtered) < 0.6 * n_t:
            print("[权重] ⚠ 匹配率 <60%：命名可能不兼容，请核对变体是否一致"
                  "（base/large/small 的维度不同，形状对不上就会被跳过）")
    return len(filtered), len(missing), len(skipped)
