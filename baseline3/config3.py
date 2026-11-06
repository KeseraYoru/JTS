# -*- coding: utf-8 -*-
"""baseline3 全局配置：换预训练起点 + 把踩过的坑变成默认值。

与 baseline1 的根本差异
----------------------
baseline1 的起点被限制在 **torchvision 的 ImageNet-1k 权重**，那是规则允许范围内最弱的一档。
复查规则第 1 条与《补充》二/五后确认：

    以 ImageNet、COCO、LAION 等通用自然图像或公开文本数据预训练的模型视为合规
    以 ImageNet、COCO、LAION 等通用非医学公开无标签数据开展自监督预训练，允许

也就是说 **ImageNet-21k 监督权重** 与 **网页级自监督权重（DINOv2，训练集 LVD-142M 来自公开网页图）**
都在允许范围内，而 baseline1 从未使用过。baseline3 就是为换这两类起点而生。

合规红线（逐条对应规则原文）
--------------------------
- 只允许非医学主数据集的开源权重；NC 许可证、gated 模型不可用；
- 禁止任何外部医学数据（含无标签）参与预训练/续训/自监督/伪标签/微调；
- 官方训练集自监督允许（补充四），测试集任何形式使用禁止（第 5 条）。
"""
import os

_THIS = os.path.dirname(os.path.abspath(__file__))

# ---------------- 路径 ----------------
DATA_ROOT = os.environ.get("CXR_DATA_ROOT", os.path.join(_THIS, "..", "cxr_first"))

# 预处理图根目录：data600（默认）或 data1024（若做过更高分辨率源）
PREP_ROOT = os.environ.get("CXR_PREP_ROOT", os.path.join(_THIS, "..", "baseline", "data600"))
# 该预处理目录对应的源边长；会写进 ckpt，推理端据此决定 draft 解码尺度
SRC_SIZE = int(os.environ.get("CXR_SRC_SIZE", "600"))

RUNS_DIR = os.path.join(_THIS, "runs")
WEIGHTS_DIR = os.path.join(_THIS, "weights")
PACKAGE_DIR = os.path.join(_THIS, "package")
for _d in (RUNS_DIR, WEIGHTS_DIR, PACKAGE_DIR):
    os.makedirs(_d, exist_ok=True)

SPLITS = ("train", "valid", "test")

# ---------------- 任务定义 ----------------
NUM_CLASSES = 10
CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CLASS_NAMES = {
    1: "纵隔心影增大", 2: "气胸", 3: "肺部实变", 4: "肺炎", 5: "肺水肿",
    6: "心影扩大", 7: "肺不张", 8: "肺部阴影", 9: "胸腔积液", 10: "无异常发现",
}

# ---------------- 模型 ----------------
ARCH = os.environ.get("CXR_ARCH", "convnext_base")
IMG_SIZE = int(os.environ.get("CXR_IMG_SIZE", "384"))
DROPOUT = 0.1

# ViewPosition 元数据（此前从未使用；同类赛事第二名用它做投影路由）
VIEW_KEYS = {"PA": 0, "AP": 1, "LATERAL": 2, "LL": 2, "RL": 2}
VIEW_DIM = 4          # 0=PA 1=AP 2=Lateral 3=未知/空

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- 训练 ----------------
EPOCHS = 12
BATCH_SIZE = 32
NUM_WORKERS = 8
USE_AMP = True

# ★ 关键修正：学习率按 batch 线性缩放。
#   baseline1 在 batch 8 上仍用 batch 32 的 3e-4，实测掉了 0.0067，一度被误读成"batch 的错"。
FULL_BATCH = 32        # 基准 batch：LR 的参考点
LR = 3e-4              # 骨干学习率（AdamW @ batch=32）
LR_SCALE = "linear"    # linear | sqrt | none，按 batch/FULL_BATCH 缩放
HEAD_LR_MULT = 10.0
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1.0    # 之前的 WARMUP_RATIO=0.05 太短，ViT 尤其需要更长 warmup
GRAD_CLIP = 5.0
LAYER_DECAY = 0.75     # ViT 逐层学习率衰减（ConvNeXt 设 1.0 即关闭）
EMA_DECAY = 0.9997
FREEZE_EPOCHS = 0

# 损失（baseline1 已验证：focal + pos_weight(cap10) 是这套数据的最优组合；
# ASL 与类别平衡采样在 batch8 混淆下均为负收益，见交接文档）
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
FOCAL_MIX = 0.7
POS_WEIGHT_BETA = 0.9999
POS_WEIGHT_CAP = 10.0

# 增强强度：mild = baseline1 原配方；strong = 向获奖方案靠拢（更大多尺度裁剪 + 更强像素扰动）
AUG = os.environ.get("CXR_AUG", "mild")

# ---------------- 推理 ----------------
AGG = "mean"
PROB_DECIMALS = 6
DEFAULT_SRC_SIZE = SRC_SIZE

# 后处理：类 10「无异常」与其余 9 类互斥（数据实证 12649 行 = 纯类10）。
# 当模型很确信"正常"时压低异常类概率 —— 同类赛事冠军的 Normal Gating，属确定性后处理。
GATE_GAMMA = float(os.environ.get("CXR_GATE_GAMMA", "0"))

# ---------------- 合规 ----------------
MODEL_SIZE_LIMIT_MB = 500
INFERENCE_TIME_LIMIT_MS = 100
TEAM_NAME = os.environ.get("CXR_TEAM", "队伍名")

# 预训练来源登记表（申报 A1~A6 直接取用）
# 硬性要求：公网直链、无需登录、许可证允许再分发、非 NC、非 gated
PRETRAIN_SOURCES = {
    "convnext_base": {
        "name": "torchvision ConvNeXt-Base（ImageNet-1k）",
        "url": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        "license": "BSD-3-Clause",
        "upstream": "ImageNet-1k（通用自然图像）",
    },
    "convnext_base_in22k": {
        # 用 "22k→1k 微调" 版本而不是纯 22k：迁移效果通常更好（timm 的 fb_in22k_ft_in1k 就是它），
        # 且体积更小（354MB vs 440MB）。两个都是 MIT，二者取其一即可。
        "name": "ConvNeXt-Base（ImageNet-22k 预训练 + ImageNet-1k 微调，官方 release）",
        "url": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth",
        "alt_url": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
        "license": "MIT",
        "upstream": "ImageNet-22k → ImageNet-1k（均为通用自然图像，非医学）",
    },
    "dinov2_vits14": {
        "name": "DINOv2 ViT-S/14（LVD-142M 自监督）",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "license": "Apache-2.0",
        "upstream": "LVD-142M（公开网页图像自监督，非医学主数据集）",
    },
    "dinov2_vitb14": {
        "name": "DINOv2 ViT-B/14（LVD-142M 自监督）",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "license": "Apache-2.0",
        "upstream": "LVD-142M（公开网页图像自监督，非医学主数据集）",
    },
}
