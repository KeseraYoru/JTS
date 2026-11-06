# -*- coding: utf-8 -*-
"""数据集与变换（纯 torchvision，不引入 albumentations）。

相比 baseline1 的两处升级
------------------------
1. **strong 增强**：baseline1 只有 RandomResizedCrop(0.85~1.0) + 轻旋转/亮度，
   而公开的胸片多标签获奖方案普遍用更大尺度的裁剪 + 更强的像素扰动 + 遮挡。
   `strong` 档把裁剪下限放到 0.65（仍不切走关键解剖结构）、加 RandAugment 级别的
   像素扰动、加大 RandomErasing。这是 baseline1 计划里被跳过的 A2 实验。
2. **返回 4 件套**：多了 ViewPosition one-hot。模型侧可选是否使用，
   不使用（view_dim=0）时行为等同 baseline1。
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from common3 import encode_labels, encode_view, prep_img_path
from config3 import (AUG, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
                     SRC_SIZE, VIEW_DIM)


# ---------------- 变换 ----------------
def _to_tensor_norm():
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def make_train_tfms(img_size, aug=None):
    """训练增强。aug: mild（baseline1 同款）| strong（向获奖配方靠拢）。"""
    aug = aug or AUG
    if aug == "mild":
        return v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1),
                                 antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.RandomRotation(degrees=10)], p=0.3),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.4),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.1),
            *_to_tensor_norm(),
            v2.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
    if aug == "strong":
        return v2.Compose([
            # 裁剪下限 0.65：胸片靠整体构图，再小会切掉肋膈角/纵隔
            v2.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25),
                                 antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.RandomRotation(degrees=12)], p=0.4),
            # 像素级扰动：胸片是灰度转 RGB，颜色通道几乎全相关，
            # 所以主要加亮度/对比度，色相只给很小的幅度
            v2.RandomApply([v2.ColorJitter(brightness=0.35, contrast=0.35,
                                           saturation=0.05, hue=0.02)], p=0.7),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
            v2.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=1.6)], p=0.2),
            *_to_tensor_norm(),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    raise ValueError(f"未知 aug 档位: {aug}（可选 mild/strong）")


def make_eval_tfms(img_size):
    """确定性评估变换（训练/推理必须完全一致，否则指标会漂）。"""
    return v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        *_to_tensor_norm(),
    ])


class ScaleBrightness:
    """TTA 用的确定性亮度缩放（模块级类，保证可 pickle）。"""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return (x * self.factor).clamp(0.0, 1.0)


def make_tta_tfms(img_size, kinds=("base", "hflip")):
    """TTA：默认只用「原图 + 水平翻转」两种。

    为什么不再用 6 路：baseline1 实测 6 路 TTA 只 +0.001，却要 6 倍推理成本；
    而多篇胸片/皮肤病获奖方案用的正是「仅 hflip」这一种，性价比最高。
    """
    tfms = []
    if "base" in kinds:
        tfms.append(v2.Compose([v2.Resize((img_size, img_size), antialias=True),
                                *_to_tensor_norm()]))
    if "hflip" in kinds:
        tfms.append(v2.Compose([v2.Resize((img_size, img_size), antialias=True),
                                v2.RandomHorizontalFlip(p=1.0),
                                *_to_tensor_norm()]))
    if "small" in kinds:
        s = int(round(img_size * 0.9))
        tfms.append(v2.Compose([v2.Resize((s, s), antialias=True),
                                v2.Resize((img_size, img_size), antialias=True),
                                *_to_tensor_norm()]))
    return tfms


# ---------------- Dataset ----------------
class ChestXrayDataset(Dataset):
    """返回 (image, dicom_id, labels, view_onehot) 四件套。

    view 用 float32 张量，便于 model 侧直接 concat；不用时把 view_dim 设为 0 即可。
    """

    def __init__(self, rows, split, transform=None, with_labels=True):
        self.rows = rows
        self.split = split
        self.transform = transform
        self.with_labels = with_labels
        self.labels = encode_labels(rows) if with_labels else None
        self.views = encode_view(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = prep_img_path(row, self.split)
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        labels = (torch.from_numpy(self.labels[idx]) if self.with_labels
                  else torch.zeros(NUM_CLASSES, dtype=torch.float32))
        view = torch.from_numpy(self.views[idx])
        return img, row["Dicom_id"], labels, view


# ---------------- 元数据辅助 ----------------
def view_stats(rows):
    """打印 ViewPosition 分布，确认元数据不是全空。"""
    v = encode_view(rows)
    names = ["PA", "AP", "Lateral", "未知"]
    return {names[i]: int(v[:, i].sum()) for i in range(v.shape[1])}


def prep_root_info():
    """训练开始前打印数据来源，防止"以为在读 1024 其实在读 600"。"""
    from config3 import PREP_ROOT
    return f"预处理图: {PREP_ROOT} | 源边长 {SRC_SIZE}px"


__all__ = ["ChestXrayDataset", "make_train_tfms", "make_eval_tfms",
           "make_tta_tfms", "view_stats", "prep_root_info"]
