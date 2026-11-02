# -*- coding: utf-8 -*-
"""损失函数（Kaggle 原版移植）：Focal Loss + 类别加权 BCE。

为什么要这么设计：本数据类别极不均衡（最罕见类只有 2.95% 正样本），
普通 BCE 会被多数类带偏；Focal Loss 让模型更关注"难样本"，
pos_weight 按 effective number of samples 给罕见类更高权重。
"""


import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """标准 Focal Loss：对易分样本降权，聚焦难样本。"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-bce)  # pt = p（正样本）或 1-p（负样本）
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class WeightedFocalBCELoss(nn.Module):
    """Kaggle 方案实际使用的损失：0.7 * Focal + 0.3 * 加权 BCE。"""

    def __init__(self, alpha=0.25, gamma=2.0, pos_weights=None, mix=0.7):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.mix = mix

    def forward(self, inputs, targets):
        return self.mix * self.focal(inputs, targets) + \
            (1 - self.mix) * self.bce(inputs, targets)


def calculate_pos_weights(label_matrix, beta=0.9999, cap=10.0):
    """按 effective number of samples（Class-Balanced Loss 论文）计算每类正样本权重。

    label_matrix: [N, 10] 的 0/1 训练标签矩阵
    返回: torch.Tensor([10])
    """
    label_matrix = np.asarray(label_matrix)
    n = label_matrix.shape[0]
    weights = []
    for i in range(label_matrix.shape[1]):
        pos_count = label_matrix[:, i].sum()
        neg_count = n - pos_count
        eff_pos = (1 - beta ** pos_count) / (1 - beta) if pos_count > 0 else 0.0
        eff_neg = (1 - beta ** neg_count) / (1 - beta) if neg_count > 0 else 0.0
        w = eff_neg / (eff_pos + 1e-8)
        weights.append(min(w, cap))
    return torch.tensor(weights, dtype=torch.float32)
