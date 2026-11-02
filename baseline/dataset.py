# -*- coding: utf-8 -*-
"""Dataset 与数据增强（从 Kaggle 亚军方案的 ChestXrayDataset 适配而来）。

主要适配点：
1. 图像路径由 CSV 的 Subject_id + Study_id + Dicom_id 三列拼出（原来是扁平目录）；
2. 标签由 Predict_class 分号字符串解析成 10 维 0/1（原来是 14 个 0/1 列）；
3. 增强 pipeline 与 Kaggle 完全一致（几何 + 光度 + dropout 三类）。
"""
import cv2
import functools
import numpy as np
import torch
from torch.utils.data import Dataset

from common import encode_labels, prep_img_path
from config import NUM_CLASSES

cv2.setNumThreads(0)  # 防止 OpenCV 抢 DataLoader worker 的 CPU

# ImageNet 归一化均值/方差（EVA-X、CheXFound 都用这套，Kaggle 原版同值）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXrayDataset(Dataset):
    """胸部 X 光数据集。

    rows:  common.read_split_csv 读出的 list[dict]
    split: "train" | "valid" | "test"
    transform: albumentations 的 A.Compose（train_tfms / val_tfms）
    prep_root: 预处理后小图的根目录
    with_labels: False 时（测试集推理）labels 返回全零占位向量

    __getitem__ 统一返回 (img, dicom_id, labels) 三件套：
    第二件永远是图像 id 字符串（推理要对齐/聚合用），第三件是 [10] 标签向量。
    注意不要改成"有标签时返回标签、没标签时返回 id"这种可变结构——
    DataLoader 的 collate 和调用方的解包会因此错位（实测踩过坑）。
    """

    def __init__(self, rows, split, transform=None, prep_root=None,
                 with_labels=True):
        self.rows = rows
        self.split = split
        self.transform = transform
        self.prep_root = prep_root
        self.with_labels = with_labels
        # 标签一次性全部编码好（比 __getitem__ 里反复解析快）
        self.labels = encode_labels(rows) if with_labels else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = prep_img_path(row, self.split, self.prep_root)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(
                f"图像不存在: {img_path}（是否忘了先跑 preprocess.py 预处理？）"
            )
        if img.ndim == 2:                      # 灰度 -> 3 通道
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:                # RGBA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # OpenCV 读出是 BGR

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.with_labels:
            labels = torch.from_numpy(self.labels[idx])
        else:
            # 推理场景：labels 用全零占位，保证三件套结构不变
            labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        return img, row["Dicom_id"], labels


def _coarse_dropout(img_size, p=0.2):
    """CoarseDropout 跨 albumentations 1.x/2.x 兼容（Kaggle 原版语义：1 个 20px 方孔）。

    albumentations 2.x 把参数改名了：max_holes/max_height/max_width（像素）
    → num_holes_range/hole_height_range/hole_width_range（图像比例），
    且旧参数在新版里是被静默忽略的（只报 warning），增强会悄悄走默认值。
    这里按大版本号选参数，保证两个版本下行为一致。
    """
    import albumentations as A

    px = 20
    major = int(A.__version__.split(".")[0])
    if major >= 2:
        frac = px / img_size
        return A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(frac, frac),
            hole_width_range=(frac, frac),
            p=p,
        )
    return A.CoarseDropout(max_holes=1, max_height=px, max_width=px, p=p)


def make_train_tfms(img_size):
    """训练增强（Kaggle 原版）：随机裁剪 + 翻转 + 旋转 + CLAHE + 亮度对比度
    + CoarseDropout + 平移缩放。"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, p=0.4),
        A.CLAHE(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        _coarse_dropout(img_size, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05,
                           rotate_limit=0, p=0.25),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def make_val_tfms(img_size):
    """验证/推理增强（Kaggle 原版）：只 resize + 归一化，不做任何随机。"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _scale_brightness_impl(image, factor=1.0, **kwargs):
    """按 factor 缩放像素亮度（裁剪回 0~255）。

    必须是模块级函数：DataLoader 的多进程 worker（macOS/新版本 Python 用
    spawn 启动）要把 dataset 连带 transform 一起 pickle 过去，
    嵌套的局部函数无法被 pickle，会把 TTA 推理搞崩。
    """
    return (image * factor).clip(0, 255).astype(image.dtype)


def scale_brightness(factor):
    """返回可 pickle 的亮度缩放 callable（functools.partial 包装模块级函数）。"""
    return functools.partial(_scale_brightness_impl, factor=factor)


def make_tta_tfms(img_size):
    """TTA 变换组（Kaggle 推理 notebook 原版）：6 种变换，推理时取平均。

    1. 原图 2. 水平翻转 3-4. 多尺度中心裁剪(1.0/1.1) 5-6. 亮度缩放(0.95/1.05)
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return [
        A.Compose([A.Resize(img_size, img_size), normalize, ToTensorV2()]),
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(img_size, img_size),
                   normalize, ToTensorV2()]),
        *[A.Compose([A.Resize(int(img_size * s), int(img_size * s)),
                     A.CenterCrop(img_size, img_size), normalize, ToTensorV2()])
          for s in (1.0, 1.1)],
        A.Compose([A.Resize(img_size, img_size),
                   A.Lambda(image=scale_brightness(0.95)),
                   normalize, ToTensorV2()]),
        A.Compose([A.Resize(img_size, img_size),
                   A.Lambda(image=scale_brightness(1.05)),
                   normalize, ToTensorV2()]),
    ]
