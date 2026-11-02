# -*- coding: utf-8 -*-
"""EVA-X base 模型（从 Kaggle 亚军 notebook `evax-recs-448-div-a.ipynb` 移植）。

EVA-X 是在 MIMIC-CXR 等大规模胸部 X 光数据上预训练的视觉 Transformer。
移植内容：
1. checkpoint_filter_fn：把官方 checkpoint 的键名映射到 timm 的 Eva 结构；
2. EVA_X：重写 forward 以支持旋转位置编码（RoPE）；
3. eva_x_base_patch16：按 Kaggle 超参构建 base 版（embed_dim=768, depth=12）；
4. EVAX_Model：包一层并把分类头改成 10 类（Kaggle 是 14 类）；
5. download_weights：自动从 HuggingFace 下载权重（内置 hf-mirror 备用地址）。

依赖：timm（pip install timm），权重文件约 344MB。
"""
import os
import sys
import urllib.request

import torch
import torch.nn as nn
from timm.models.eva import Eva

from config import (
    EVAX_IMG_SIZE,
    EVAX_WEIGHT_FILE,
    EVAX_WEIGHT_URL,
    HF_ENDPOINT,
    NUM_CLASSES,
    WEIGHTS_DIR,
)


# ---------------- checkpoint 键名映射（Kaggle 原版） ----------------

def checkpoint_filter_fn(state_dict, model, interpolation="bicubic", antialias=True):
    """把官方 checkpoint 的键名/形状适配到 timm 的 Eva 模型。

    主要处理：OpenCLIP 前缀剥离、patch embedding 权重形状转换、
    位置编码重采样（换输入尺寸时）、MLP/Attention 键名改名、MIM 预训练头剔除。
    """
    from timm.layers import resample_abs_pos_embed, resample_patch_embed

    out_dict = {}
    state_dict = state_dict.get("model_ema", state_dict)
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("module", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)
    if "visual.trunk.pos_embed" in state_dict:
        prefix = "visual.trunk."
    elif "visual.pos_embed" in state_dict:
        prefix = "visual."
    else:
        prefix = ""
    mim_weights = prefix + "mask_token" in state_dict
    no_qkv = prefix + "blocks.0.attn.q_proj.weight" in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if "rope" in k:
            continue

        if k == "patch_embed.proj.weight" and \
                v.shape[2:] != model.patch_embed.proj.weight.shape[2:]:
            v = resample_patch_embed(
                v,
                new_size=model.patch_embed.patch_size,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = 0 if getattr(model, "no_embed_class", False) \
                else getattr(model, "num_prefix_tokens", 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace("mlp.ffn_ln", "mlp.norm")
        k = k.replace("attn.inner_attn_ln", "attn.norm")
        k = k.replace("mlp.w12", "mlp.fc1")
        k = k.replace("mlp.w1", "mlp.fc1_g")
        k = k.replace("mlp.w2", "mlp.fc1_x")
        k = k.replace("mlp.w3", "mlp.fc2")
        if no_qkv:
            k = k.replace("q_bias", "q_proj.bias")
            k = k.replace("v_bias", "v_proj.bias")

        if mim_weights and k in ("mask_token", "lm_head.weight",
                                 "lm_head.bias", "norm.weight", "norm.bias"):
            if k == "norm.weight" or k == "norm.bias":
                k = k.replace("norm", "fc_norm")
            else:
                continue

        out_dict[k] = v

    return out_dict


# ---------------- 模型定义（Kaggle 原版） ----------------

class EVA_X(Eva):
    """重写 forward 以支持旋转位置编码（RoPE）。"""

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) \
                if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def eva_x_base_patch16(pretrained=None, img_size=EVAX_IMG_SIZE):
    """按 Kaggle 超参构建 EVA-X base（patch16, embed_dim=768, depth=12, 12 heads）。"""
    model = EVA_X(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),  # 224/16，Kaggle 原值
    )
    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        filtered = checkpoint_filter_fn(ckpt, model)
        msg = model.load_state_dict(filtered, strict=False)
        print(f"[EVA-X] 权重加载情况: {msg}")
    return model


class EVAX_Model(nn.Module):
    """Kaggle 的 EVAX_Model：EVA-X backbone + 10 类线性分类头。"""

    def __init__(self, num_classes=NUM_CLASSES, pretrained_path=None,
                 img_size=EVAX_IMG_SIZE):
        super().__init__()
        self.backbone = eva_x_base_patch16(pretrained=pretrained_path,
                                           img_size=img_size)
        self.backbone.head = nn.Linear(in_features=768, out_features=num_classes,
                                       bias=True)
        for p in self.backbone.head.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# ---------------- 权重下载 ----------------

def download_weights(weights_dir=WEIGHTS_DIR):
    """下载 EVA-X 预训练权重到 weights/ 目录；已存在则直接返回路径。

    国内网络默认地址打不开时，设置 CXR_HF_ENDPOINT=https://hf-mirror.com 再跑。
    """
    os.makedirs(weights_dir, exist_ok=True)
    dst = os.path.join(weights_dir, EVAX_WEIGHT_FILE)
    if os.path.exists(dst):
        print(f"[EVA-X] 权重已存在: {dst}")
        return dst

    urls = [EVAX_WEIGHT_URL]
    if "huggingface.co" in EVAX_WEIGHT_URL and HF_ENDPOINT != "https://huggingface.co":
        urls.append(EVAX_WEIGHT_URL.replace("https://huggingface.co", HF_ENDPOINT))

    last_err = None
    for url in urls:
        try:
            print(f"[EVA-X] 下载权重（约 344MB）: {url}")
            _download_with_progress(url, dst)
            print(f"[EVA-X] 权重已保存: {dst}")
            return dst
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[EVA-X] 下载失败: {e}")
    raise RuntimeError(
        "EVA-X 权重下载失败。请手动下载以下文件并放到 "
        f"{dst} ：\n  {EVAX_WEIGHT_URL}\n"
        "（国内可用镜像：把 URL 里的 huggingface.co 换成 hf-mirror.com）\n"
        f"原始错误: {last_err}"
    )


def _download_with_progress(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                sys.stdout.write(
                    f"\r  进度 {downloaded / 1024 / 1024:.0f}MB / "
                    f"{total / 1024 / 1024:.0f}MB"
                )
                sys.stdout.flush()
    print()


# ---------------- 工厂函数 ----------------

def build_evax(num_classes=NUM_CLASSES, pretrained=True, weights_dir=WEIGHTS_DIR):
    """构建 EVAX_Model；pretrained=True 时自动下载/加载 MIMIC 预训练权重。"""
    ckpt_path = None
    if pretrained:
        ckpt_path = download_weights(weights_dir)
    return EVAX_Model(num_classes=num_classes, pretrained_path=ckpt_path)


def build_evax_optimizer(model, lr_head=8e-5, weight_decay=1e-4):
    """Kaggle 原版的 layer-wise 学习率：越深的 transformer block 学习率越大，
    分类头最大。浅层特征更通用，小步慢改；深层更接近任务，大步快改。"""
    b = getattr(model, "module", model).backbone   # 兼容 DataParallel 包装
    groups = [
        {"params": b.patch_embed.parameters(), "lr": 1e-6, "weight_decay": 1e-6},
        {"params": b.blocks[0:4].parameters(), "lr": 1e-6, "weight_decay": weight_decay},
        {"params": b.blocks[4:6].parameters(), "lr": 5e-6, "weight_decay": weight_decay},
        {"params": b.blocks[6:8].parameters(), "lr": 1e-5, "weight_decay": weight_decay},
        {"params": b.blocks[8:10].parameters(), "lr": 3e-5, "weight_decay": weight_decay},
        {"params": b.blocks[10:].parameters(), "lr": 5e-5, "weight_decay": weight_decay},
        {"params": b.fc_norm.parameters(), "lr": 5e-5, "weight_decay": weight_decay},
        {"params": b.head.parameters(), "lr": lr_head, "weight_decay": 0},
    ]
    # rope 模块不是所有 timm 版本都有，有就按 Kaggle 原版给 1e-6
    rope = getattr(b, "rope", None)
    if rope is not None:
        groups.insert(1, {"params": rope.parameters(), "lr": 1e-6,
                          "weight_decay": 1e-6})
    return torch.optim.AdamW(groups)
