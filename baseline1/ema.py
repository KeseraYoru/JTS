# -*- coding: utf-8 -*-
"""EMA（指数移动平均，Kaggle 原版移植）。

直觉：模型权重在训练中会来回抖动，EMA 把最近若干步的权重做加权平均，
相当于给成绩单做"移动平均"，通常能稳定提升验证集表现。
"""
import copy

import torch


def get_model(module):
    """剥掉 DataParallel/DDP 包装，拿到真正的模型。"""
    return module.module if hasattr(module, "module") else module


class ModelEMA:
    """对模型参数（含 buffer）做指数移动平均。

    decay=0.9997 表示**每步**保留 99.97% 的旧值、混入 0.03% 的新值。
    等价于对最近约 1/(1-decay)=3333 步的权重做加权平均（≈3.5 个 epoch）。

    ⚠ 两个必须牢记的点：
    1. `update()` 必须**每个训练 step 调一次**。如果只在每个 epoch 末尾调，
       15 轮总共只混入 15×0.03% = 0.45% 的新权重 → EMA 几乎完全是初始权重，
       验证集评估的其实是"训练前的模型"（2026-10-30 踩过，AUC 卡在 0.58 不动）。
    2. decay 有 warmup：训练早期 num_updates 很小时用较小的 decay 快速跟上，
       否则第 1 个 epoch 的 EMA 里有 75% 是初始权重（0.9997^948），
       每轮打印的 AUC 会严重低估模型。
    """

    def __init__(self, model, decay=0.9997, device=None, warmup=True):
        self.base_decay = decay
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.device = device
        # 深拷贝一份"干净"的模型专门存 EMA 权重
        self.ema = copy.deepcopy(get_model(model)).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """每个训练 step 调用一次（不要放在 epoch 循环里）。"""
        self.num_updates += 1
        decay = self.base_decay
        if self.warmup:
            # 第 1 步 decay=0.5（几乎直接复制），第 948 步 0.9989，
            # 第 3333 步起封顶到 base_decay。这样早期 EMA 也跟得上。
            decay = min(decay, self.num_updates / (self.num_updates + 1.0))
        self.decay = decay

        src_state = get_model(model).state_dict()
        vals, srcs = [], []
        for k, v in self.ema.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(src_state[k].to(v.device))   # 整数 buffer 直接覆盖
                continue
            vals.append(v)
            srcs.append(src_state[k].detach().to(v.device))
        if vals:
            # in-place + foreach：避免每步创建临时张量，88M 参数下快数倍
            torch._foreach_mul_(vals, decay)
            torch._foreach_add_(vals, srcs, alpha=1.0 - decay)

    def state_dict(self):
        return self.ema.state_dict()
