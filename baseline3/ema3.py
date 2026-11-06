# -*- coding: utf-8 -*-
"""EMA（指数移动平均）：每个 **step** 更新一次，带 warmup。

这是 baseline1 用 4 次 15 轮训练换来的教训：
`update()` 若写在 epoch 循环里，15 轮只混入 0.45% 的新权重，
最终评估的其实是"训练前的模型"，loss 一路下降但 AUC 不动。
所以本文件把更新调用**封装进 train_one_epoch**，不给写错的机会。
"""
import copy

import torch

from model3 import get_inner


class ModelEMA:
    def __init__(self, model, decay=0.9997, device=None, warmup=True):
        self.base_decay = decay
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.ema = copy.deepcopy(get_inner(model)).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """每个训练 step 调一次。"""
        self.num_updates += 1
        decay = self.base_decay
        if self.warmup:
            decay = min(decay, self.num_updates / (self.num_updates + 1.0))
        self.decay = decay

        src = get_inner(model).state_dict()
        vals, srcs = [], []
        for k, v in self.ema.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(src[k].to(v.device))
                continue
            vals.append(v)
            srcs.append(src[k].detach().to(v.device))
        if vals:
            torch._foreach_mul_(vals, decay)
            torch._foreach_add_(vals, srcs, alpha=1.0 - decay)

    def state_dict(self):
        return self.ema.state_dict()
