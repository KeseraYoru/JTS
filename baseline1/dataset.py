# -*- coding: utf-8 -*-
"""数据集与数据变换（纯 torchvision 实现，不依赖 albumentations/opencv）。

为什么换掉 albumentations：
- 复赛容器只保证 torch + torchvision，少一个依赖就少一处复现风险；
- 之前踩过「albumentations 2.x 静默忽略参数」的坑，torchvision 的 API 更稳定。

约定（沿用 baseline/ 的重要设计）：`__getitem__` 永远返回 **三件套**
`(image, dicom_id, labels)`。少一个都会导致推理时拿不到 id 或把标签当 id。
"""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from common import encode_labels, prep_img_path
from config import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def _to_tensor_norm():
    """PIL/uint8 -> float32 [0,1] -> ImageNet 归一化。"""
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def make_train_tfms(img_size):
    """训练增强：随机裁剪/翻转/小角度旋转/亮度对比度/轻微模糊/随机遮挡。

    注意尺度下限用 0.85（不用 0.3~0.5 那种激进裁剪）——胸片靠整体构图判断，
    裁掉太多会把关键区域切走。
    """
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


def make_eval_tfms(img_size):
    """验证/推理变换：只做确定性 resize + 归一化（无任何随机算子）。"""
    return v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        *_to_tensor_norm(),
    ])


class ScaleBrightness:
    """确定性亮度缩放（TTA 用）。

    必须是**模块级类**而不是嵌套函数：macOS 的 DataLoader 用 spawn 启动，
    要把变换一起 pickle 过去，局部函数会直接崩（踩过这个坑）。
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return (x * self.factor).clamp(0.0, 1.0)


def make_tta_tfms(img_size):
    """TTA：6 种**确定性**变换，推理时取平均（规则第 6 条明确允许 TTA）。

    1. 原图  2. 水平翻转  3-4. 多尺度（放大/缩小后中心裁剪）
    5-6. 亮度 0.95 / 1.05

    全部不含随机算子——规则要求推理时关闭所有随机增强。
    """
    base = [
        v2.Resize((img_size, img_size), antialias=True),
        *_to_tensor_norm(),
    ]
    bigger = int(round(img_size * 1.1))
    smaller = int(round(img_size * 0.9))
    return [
        v2.Compose(base),
        v2.Compose([v2.Resize((bigger, bigger), antialias=True),
                    v2.CenterCrop(img_size), *_to_tensor_norm(),
                    v2.RandomHorizontalFlip(p=1.0)]),
        v2.Compose([v2.Resize((bigger, bigger), antialias=True),
                    v2.CenterCrop(img_size), *_to_tensor_norm()]),
        v2.Compose([v2.Resize((smaller, smaller), antialias=True),
                    v2.Resize((img_size, img_size), antialias=True),
                    *_to_tensor_norm()]),
        v2.Compose([v2.Resize((img_size, img_size), antialias=True),
                    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                    ScaleBrightness(0.95),
                    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
        v2.Compose([v2.Resize((img_size, img_size), antialias=True),
                    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                    ScaleBrightness(1.05),
                    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    ]


def make_ssl_tfms(img_size):
    """SimCLR 双视图变换（自监督预训练用，只在训练阶段使用）。"""
    view = v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.35, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.05)], p=0.6),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3),
        *_to_tensor_norm(),
    ])
    return view


class ChestXrayDataset(Dataset):
    """胸部 X 光多标签数据集。

    rows:      read_split_csv 的结果
    split:     "train" | "valid" | "test"
    transform: v2.Compose（训练/验证/TTA 之一）
    with_labels: 推理测试集时为 False（返回零向量占位，保持三件套结构）
    """

    def __init__(self, rows, split, transform=None, with_labels=True):
        self.rows = rows
        self.split = split
        self.transform = transform
        self.with_labels = with_labels
        self.labels = encode_labels(rows) if with_labels else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = prep_img_path(row, self.split)
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.with_labels:
            labels = torch.from_numpy(self.labels[idx])
        else:
            labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        return img, row["Dicom_id"], labels


class SSLTwoView(Dataset):
    """SimCLR 用的两视图数据集：同一张图返回两个独立增强视图。"""

    def __init__(self, rows, split, img_size):
        self.rows = rows
        self.split = split
        self.view = make_ssl_tfms(img_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path = prep_img_path(self.rows[idx], self.split)
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.view(img), self.view(img)
