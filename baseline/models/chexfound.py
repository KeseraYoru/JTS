# -*- coding: utf-8 -*-
"""CheXFound 模型（从 Kaggle 亚军 notebook `all-data-chexfound-recs-div-a_*.ipynb` 移植）。

CheXFound 是在大规模 X 光上自监督预训练的 DINOv2 式 ViT，官方附带一个
GLoRIA 线性分类头（在 CheXpert 40 类上训练）。Kaggle 方案只微调这个分类头，
backbone 全程冻结。

移植内容：
1. 加载官方 teacher_checkpoint.pth（键名 backbone.xxx -> xxx 的重映射）；
2. 按 results_eval_linear.json 里的最佳分类器配置重建 GLoRIA 头；
3. 把分类头从 40 类改造成 10 类（Kaggle 是改成 14 类）；
4. 冻结 backbone，只训 glori 头。

使用前需要手动准备两部分文件（详见 README 第 2 节）：
  third_party/CheXFound/          # git clone https://github.com/RPIDIAL/CheXFound.git
  third_party/CheXFound_Model/    # 官方权重：config.yaml / teacher_checkpoint.pth /
                                  # glori.pth / results_eval_linear.json
依赖：fvcore / open_clip_torch / omegaconf / scikit-multilearn / torchmetrics /
iopath / pandas / transformers / scikit-image（跑 `python3 check_chexfound.py`
可一键检查是否装齐；xformers 不用装，代码会自动回退原生 attention）。
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

from config import (
    CHEXFOUND_MODEL_DIR,
    CHEXFOUND_ORIG_NUM_CLASSES,
    CHEXFOUND_REPO,
    NUM_CLASSES,
)


def _require_chexfound_files():
    """检查第三方代码和权重是否就位，缺什么就给明确提示。"""
    problems = []
    if not os.path.isdir(CHEXFOUND_REPO):
        problems.append(f"缺少 CheXFound 代码目录: {CHEXFOUND_REPO}")
    for fname in ("config.yaml", "teacher_checkpoint.pth", "glori.pth",
                  "results_eval_linear.json"):
        p = os.path.join(CHEXFOUND_MODEL_DIR, fname)
        if not os.path.exists(p):
            problems.append(f"缺少权重/配置文件: {p}")
    if problems:
        raise FileNotFoundError(
            "CheXFound 依赖未就位：\n  " + "\n  ".join(problems) +
            "\n请按 README 第 2 节准备（git clone + 下载官方权重）。"
        )


def _make_args():
    """Kaggle 原版的 argparse 默认参数（用 parse_known_args 吃掉 Jupyter 参数）。"""
    parser = argparse.ArgumentParser()
    parser.set_defaults(
        config_file=os.path.join(CHEXFOUND_MODEL_DIR, "config.yaml"),
        pretrained_weights=None,
        output_dir=os.path.join(CHEXFOUND_MODEL_DIR, "example"),
        opts=[],
        image_size=512,
        patch_size=16,
        n_register_tokens=4,
        n_last_blocks=4,
        return_class_token=True,
        num_classes=CHEXFOUND_ORIG_NUM_CLASSES,  # 官方 glori 头是 40 类
        num_heads=8,
    )
    args, _ = parser.parse_known_args()
    return args


def _load_backbone(args):
    """构建 backbone 并加载 teacher_checkpoint.pth（Kaggle 键名重映射原版）。"""
    from chexfound.eval.setup import setup_and_build_model

    base_model, _ = setup_and_build_model(args)
    state_dict = torch.load(
        os.path.join(CHEXFOUND_MODEL_DIR, "teacher_checkpoint.pth"),
        map_location="cpu", weights_only=False,
    )["teacher"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone"):
            ls = k.split(".")
            if "blocks" in k:
                new_k = ".".join([ls[1], *ls[3:]])
            else:
                new_k = ".".join(ls[1:])
        else:
            new_k = k
        new_state_dict[new_k] = v
    base_model.load_state_dict(new_state_dict, strict=False)
    return base_model


def _load_glori(base_model, args):
    """按官方 results_eval_linear.json 的最佳配置重建 GLoRIA 头并加载权重。"""
    from chexfound.eval.classification.utils import setup_glori
    from chexfound.eval.utils import extract_hyperparameters_from_model
    from fvcore.common.checkpoint import Checkpointer

    with open(os.path.join(CHEXFOUND_MODEL_DIR, "results_eval_linear.json"),
              "r", encoding="utf-8") as f:
        content = f.read().split("\n")[-3]
        data = json.loads(content)
    best_classifier_str = data["best_classifier"]["name"]
    hyperparameters = extract_hyperparameters_from_model(best_classifier_str)
    learning_rate = hyperparameters["lr"]
    avgpool = hyperparameters["avgpool"]
    block = hyperparameters["blocks"]

    device = next(base_model.parameters()).device
    sample_input = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        sample_output = base_model.get_intermediate_layers(
            sample_input, n=args.n_last_blocks, return_class_token=True
        )

    glori, _ = setup_glori(
        sample_output=sample_output,
        n_last_blocks_list=block,
        learning_rates=learning_rate,
        avgpools=avgpool,
        num_classes=args.num_classes,
        multiview=False,
        decoder_dim=768,
        cat_cls=False,
    )
    checkpointer = Checkpointer(glori)
    checkpointer.load(os.path.join(CHEXFOUND_MODEL_DIR, "glori.pth"))
    return glori


def _resize_glori_classes(glori, target_classes=NUM_CLASSES,
                          orig_classes=CHEXFOUND_ORIG_NUM_CLASSES):
    """把 GLoRIA 分类头从 orig_classes(40) 类改造成 target_classes(10) 类。

    Kaggle 原版逻辑：改 decoder.num_classes，替换所有长度为 40 的 1 维
    参数/缓冲（新头随机初始化，训练几轮就会收敛）。
    """
    device = next(glori.parameters()).device
    glori_module = getattr(glori, "module", glori)
    classifier_key = next(iter(glori_module.classifiers_dict.keys()))
    classifier = glori_module.classifiers_dict[classifier_key]
    decoder = getattr(classifier, "decoder", None)
    if decoder is None:
        raise RuntimeError("未找到 decoder，无法改造类别数，请检查 glori 结构")

    decoder.num_classes = target_classes
    if hasattr(decoder, "duplicate_pooling_bias"):
        decoder.duplicate_pooling_bias = nn.Parameter(
            torch.zeros(target_classes, dtype=torch.float32, device=device)
        )

    replaced = []
    for name, param in list(decoder.named_parameters()):
        if param.dim() == 1 and param.shape[0] == orig_classes:
            parent = decoder
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1],
                    nn.Parameter(torch.zeros(target_classes, dtype=param.dtype,
                                             device=device)))
            replaced.append(name)
    for name, buf in list(decoder.named_buffers()):
        if buf is None:
            continue
        if buf.ndim == 1 and buf.shape[0] == orig_classes:
            parent = decoder
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent.register_buffer(
                parts[-1],
                torch.zeros(target_classes, dtype=buf.dtype, device=device),
            )
            replaced.append("buffer:" + name)
    print(f"[CheXFound] 分类头 {orig_classes} -> {target_classes} 类，"
          f"替换: {replaced}")
    return glori


class CheXFoundWithGLoRIHead(nn.Module):
    """Kaggle 原版包装：backbone 出中间层特征 -> glori 头出 logits。"""

    def __init__(self, backbone, glori_module, args):
        super().__init__()
        self.backbone = backbone
        self.glori = glori_module
        self.args = args

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(
            x, n=self.args.n_last_blocks,
            return_class_token=self.args.return_class_token,
        )
        logits_dict = self.glori([feats], return_attention=False)
        key = list(logits_dict.keys())[0]
        return logits_dict[key]


def _suppress_library_distributed():
    """把 chexfound 库的 distributed.enable 打成 no-op（根因修复）。

    库的 default_setup（chexfound/utils/config.py）会无条件
    init_process_group(NCCL)；随后 setup_glori 发现进程组已启用，就把 glori 头
    包进 DDP（find_unused_parameters=False）——它与我们 40→10 类的头改造冲突，
    第 2 个 step 必崩（详见 build_chexfound 内注释，2026-10-22 服务器实测）。
    多卡并行由 train.py 的 DataParallel 全权负责，根本不需要进程组，
    所以从源头禁止初始化：没有进程组 → is_enabled() 为 False → 库不会包装 DDP。
    utils/config.py 是 `import chexfound.distributed as distributed`，
    改模块属性对它可见；get_global_rank() 未初始化时安全返回 0，不受影响。
    """
    import chexfound.distributed as cf_dist
    cf_dist.enable = lambda **kwargs: None


def build_chexfound(num_classes=NUM_CLASSES, freeze_backbone=True,
                    repo_path=CHEXFOUND_REPO):
    """构建 CheXFoundWithGLoRIHead。freeze_backbone=True 时只训练 glori 头。"""
    _require_chexfound_files()
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # 先禁止库初始化 NCCL 进程组（必须在任何 chexfound 导入/调用之前）
    _suppress_library_distributed()

    args = _make_args()
    base_model = _load_backbone(args)
    glori = _load_glori(base_model, args)

    # 双保险（正常流程下上面已从源头禁止，这里是空操作）：
    # 万一将来库改了行为还是包装了 DDP，这里拆掉；进程组若被初始化过也一并清掉。
    if isinstance(glori, torch.nn.parallel.DistributedDataParallel):
        glori = glori.module
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

    glori = _resize_glori_classes(glori, target_classes=num_classes)

    model = CheXFoundWithGLoRIHead(backbone=base_model, glori_module=glori,
                                   args=args)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    for p in model.glori.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[CheXFound] 可训练参数 {n_train / 1e6:.1f}M / 总参数 "
          f"{n_total / 1e6:.1f}M（backbone 已冻结）")
    return model
