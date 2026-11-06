# -*- coding: utf-8 -*-
"""损失函数：加权 Focal + 类别权重（baseline1 已验证的最优组合）。

为什么不再默认开 ASL
------------------
baseline1 在 batch8 + LR 未缩放的条件下测过 ASL（0.7885 vs 同条件 0.7930，−0.0045），
但那次对比受两个混淆变量影响。本文件仍保留 ASL 实现，方便在**正确的 batch/LR**
下重测一次；默认关闭。

mixup/cutmix 支持：标签变成软标签（[0,1] 连续），Focal 与 BCE 都能直接吃软标签，
不需要换损失函数 —— 这是多标签任务里 mixup 的正确用法。
"""
import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class WeightedFocalBCELoss(nn.Module):
    """mix × Focal + (1-mix) × 加权 BCE。"""

    def __init__(self, alpha=0.25, gamma=2.0, pos_weights=None, mix=0.7):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.mix = mix

    def forward(self, inputs, targets):
        return (self.mix * self.focal(inputs, targets)
                + (1 - self.mix) * self.bce(inputs, targets))


class AsymmetricLoss(nn.Module):
    """ASL（保留以便在正确 batch/LR 下复测；默认不用）。"""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0 and self.gamma_neg != self.gamma_pos:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss = (targets * torch.log(xs_pos.clamp(min=self.eps))
                + (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps)))
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow((1.0 - pt).clamp(min=0.0), gamma)
        return -loss.mean()


def calculate_pos_weights(label_matrix, beta=0.9999, cap=10.0):
    """effective number of samples 计算每类正样本权重。"""
    label_matrix = np.asarray(label_matrix)
    n = label_matrix.shape[0]
    weights = []
    for i in range(label_matrix.shape[1]):
        pos = label_matrix[:, i].sum()
        neg = n - pos
        eff_pos = (1 - beta ** pos) / (1 - beta) if pos > 0 else 0.0
        eff_neg = (1 - beta ** neg) / (1 - beta) if neg > 0 else 0.0
        weights.append(min(eff_neg / (eff_pos + 1e-8), cap))
    return torch.tensor(weights, dtype=torch.float32)


def mixup_batch(imgs, labels, alpha=0.2, prob=0.5):
    """多标签 mixup：lam 混合图像与软标签。

    为什么这对 AUC 有帮助：它等价于在样本之间做线性插值，抑制模型记忆训练集，
    对罕见类尤其友好（相当于给每个正样本造出无限多个"邻居"）。
    注意：只允许在**训练**使用，推理时完全关闭（规则第 4 条）。
    """
    if alpha <= 0 or np.random.rand() > prob or imgs.shape[0] < 2:
        return imgs, labels
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.shape[0], device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], lam * labels + (1 - lam) * labels[idx]
