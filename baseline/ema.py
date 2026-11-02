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

    decay=0.9997 表示每步保留 99.97% 的旧值、混入 0.03% 的新值。
    """

    def __init__(self, model, decay=0.9997, device=None):
        self.decay = decay
        self.device = device
        # 深拷贝一份"干净"的模型专门存 EMA 权重
        self.ema = copy.deepcopy(get_model(model)).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src_state = get_model(model).state_dict()
        for k, v in self.ema.state_dict().items():
            model_v = src_state[k].detach().to(v.device)
            v.copy_(v * self.decay + (1.0 - self.decay) * model_v)

    def state_dict(self):
        return self.ema.state_dict()
