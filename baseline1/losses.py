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


class AsymmetricLoss(nn.Module):
    """ASL（Asymmetric Loss for Multi-Label Classification）。

    为什么它能帮到罕见类
    -------------------
    多标签数据里 99% 的 (图, 类) 组合是负样本。Focal 对所有错误"一视同仁"地降权，
    ASL 则**非对称**地处理：
      - 正样本（罕见类的那 894 个）：gamma_pos 通常取 0，不做降权 —— 每个正样本都很珍贵；
      - 负样本（海量的"没这个病"）：gamma_neg 取 4 左右，**大幅降权**，
        并且先把概率往 0 方向平移一个 clip（默认 0.05），把"明显是负样本"的直接压到 0 权重。
    效果是：梯度几乎全来自"难负样本 + 所有正样本"，罕见类的信号不会被淹没。

    与 pos_weight 的区别（A6 已验证 pos_weight 无效，正因为方向不同）：
      pos_weight 只放大正样本的惩罚倍数，**分母里的海量负样本仍在参与**；
      ASL 是把负样本的权重真正压到接近 0，等于把"信噪比"提上去了。

    参考：Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification" (ICCV 2021).
    """

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8,
                 reduction="mean"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.symmetric = gamma_neg == gamma_pos

    def forward(self, logits, targets):
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # 负样本概率平移：把"确信是负"的压到 0，进一步减少它们占用的梯度
        if self.clip > 0 and not self.symmetric:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow((1.0 - pt).clamp(min=0.0), one_sided_gamma)

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


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
