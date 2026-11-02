# -*- coding: utf-8 -*-
"""模型工厂：按名字构建模型/优化器，train.py 和 predict.py 都从这里入口。"""
from config import CHEXFOUND_LR, NUM_CLASSES, WEIGHT_DECAY


def build_model(name, num_classes=NUM_CLASSES, pretrained=True):
    """name: "evax" | "chexfound\""""
    name = name.lower()
    if name == "evax":
        from models.evax import build_evax
        return build_evax(num_classes=num_classes, pretrained=pretrained)
    if name == "chexfound":
        from models.chexfound import build_chexfound
        return build_chexfound(num_classes=num_classes)
    raise ValueError(f"未知模型: {name}（可选 evax / chexfound）")


def build_optimizer(name, model, **kw):
    """按模型返回对应的优化器配置。

    evax:      Kaggle 原版 layer-wise 学习率（深层大、浅层小）
    chexfound: 只训 glori 头，统一小学习率 AdamW
    """
    import torch

    # train.py 多卡时会把模型包一层 DataParallel，包装后 model.backbone 等
    # 属性访问不到，先剥掉包装（参数是同一批张量，优化器照常生效）
    model = getattr(model, "module", model)

    name = name.lower()
    if name == "evax":
        from models.evax import build_evax_optimizer
        return build_evax_optimizer(
            model,
            lr_head=kw.get("lr_head", 8e-5),
            weight_decay=kw.get("weight_decay", WEIGHT_DECAY),
        )
    if name == "chexfound":
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable,
            lr=kw.get("lr", CHEXFOUND_LR),
            weight_decay=kw.get("weight_decay", WEIGHT_DECAY),
        )
    raise ValueError(f"未知模型: {name}（可选 evax / chexfound）")


def model_img_size(name):
    """各模型的输入分辨率（必须与预处理/增强/推理一致）。"""
    name = name.lower()
    if name == "evax":
        from config import EVAX_IMG_SIZE
        return EVAX_IMG_SIZE
    if name == "chexfound":
        from config import CHEXFOUND_IMG_SIZE
        return CHEXFOUND_IMG_SIZE
    raise ValueError(f"未知模型: {name}（可选 evax / chexfound）")


def default_batch_size(name):
    """各模型在 T4 16GB 上的默认 batch size。"""
    name = name.lower()
    if name == "evax":
        from config import BATCH_SIZE
        return BATCH_SIZE
    if name == "chexfound":
        from config import CHEXFOUND_BATCH_SIZE
        return CHEXFOUND_BATCH_SIZE
    raise ValueError(f"未知模型: {name}（可选 evax / chexfound）")
