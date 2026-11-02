# -*- coding: utf-8 -*-
"""全局配置：路径、骨干选择、训练超参、推理与提交口径、合规参数。

设计原则（与 baseline/ 的关键区别）：
1. **只用 torch + torchvision**（复赛容器只保证这两个），不依赖 timm / albumentations /
   opencv / fvcore / transformers；
2. 预训练权重全部来自 **torchvision 的 ImageNet 权重**——规则第 1 条允许「以 ImageNet 等
   通用自然图像预训练」的模型，医学影像预训练权重（EVA-X/CheXFound/RETFound 等）禁止；
3. 所有会影响结果的超参集中在这里，调实验只改这个文件。
"""
import os

# ---------------- 路径 ----------------
_THIS = os.path.dirname(os.path.abspath(__file__))

# 原始数据（含 3 个标注 CSV 与图片）
DATA_ROOT = os.environ.get(
    "CXR_DATA_ROOT", os.path.join(_THIS, "..", "cxr_first"))
# 预处理后的小图（与 baseline/ 共用同一份 data600，无需重跑）
PREP_ROOT = os.environ.get(
    "CXR_PREP_ROOT", os.path.join(_THIS, "..", "baseline", "data600"))
# 训练产物（ckpt / 日志）
RUNS_DIR = os.path.join(_THIS, "runs")
# torchvision 预训练权重缓存（首次自动下载到 torch hub 缓存，这里只放说明）
WEIGHTS_DIR = os.path.join(_THIS, "weights")
# 提交包输出目录
PACKAGE_DIR = os.path.join(_THIS, "package")

for _d in (RUNS_DIR, WEIGHTS_DIR, PACKAGE_DIR):
    os.makedirs(_d, exist_ok=True)

SPLITS = ("train", "valid", "test")

# ---------------- 预处理 ----------------
# 输出 600x600：比任何候选骨干的输入尺寸（384/512）都大，留出后续试更高分辨率的余地。
# 默认与 baseline/ 共用同一份 data600（已跑好 33984 张，无需重做）。
PREP_SIZE = 600
PREP_QUALITY = 95

# ---------------- 任务定义 ----------------
NUM_CLASSES = 10
# 题面类别 id（1~10）与向量下标的对应顺序，不要改
CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CLASS_NAMES = {
    1: "纵隔心影增大", 2: "气胸", 3: "肺部实变", 4: "肺炎", 5: "肺水肿",
    6: "心影扩大", 7: "肺不张", 8: "肺部阴影", 9: "胸腔积液", 10: "无异常发现",
}

# ---------------- 模型（合规选择） ----------------
# 可选（均为 torchvision 内置、ImageNet-1k 预训练、BSD 许可证）：
#   convnext_base      88.6M 参数，fp32 约 354MB  ← 默认，500MB 限内最强单模型
#   convnext_small     50.2M，约 200MB（更快，可与 efficientnet_v2_m 集成）
#   efficientnet_v2_m  54.1M，约 217MB
#   swin_b             87.8M，约 352MB
#   vit_b_16           86.6M，约 345MB
BACKBONE = os.environ.get("CXR_BACKBONE", "convnext_base")
IMG_SIZE = int(os.environ.get("CXR_IMG_SIZE", "384"))
DROPOUT = 0.1

# ImageNet 归一化统计量（torchvision 权重训练时用的，推理必须一致）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- 训练 ----------------
EPOCHS = 12
BATCH_SIZE = 32
NUM_WORKERS = 8
USE_AMP = True

LR = 3e-4              # 骨干学习率（AdamW）
HEAD_LR_MULT = 10.0    # 分类头用 LR × 该倍数
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05    # cosine 预热比例
FREEZE_EPOCHS = 0      # >0 时前 N 轮冻结骨干（只训头），之后解冻
GRAD_CLIP = 5.0

EMA_DECAY = 0.9997

# 损失（沿用 Kaggle 方案验证过的配方）
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
FOCAL_MIX = 0.7        # mix×Focal + (1-mix)×加权 BCE
POS_WEIGHT_BETA = 0.9999
POS_WEIGHT_CAP = 10.0

# ---------------- 自监督预训练（规则补充四明确允许，仅用官方训练集） ----------------
SSL_ENABLED = os.environ.get("CXR_SSL", "0") == "1"
SSL_EPOCHS = 20
SSL_BATCH_SIZE = 128
SSL_LR = 1e-3
SSL_BASE_LR = 1e-4
SSL_TEMPERATURE = 0.2
SSL_OUT = os.path.join(RUNS_DIR, "ssl", "ssl_backbone.pth")

# ---------------- 推理 ----------------
TTA = True
AGG = "mean"           # study 级聚合方式：mean | max（患者平滑，规则补充七允许）
PROB_DECIMALS = 6      # 提交概率小数位（避免并列拉低 AUC）

# ---------------- 提交 ----------------
SUBMIT_ALL_CLASSES = True   # Macro AUC 需要完整 10 类概率矩阵
SUBMIT_THRESHOLD = 0.5      # 不交全类时只交 >= 该阈值的类

# ---------------- 合规（规则第 7/8 条） ----------------
MODEL_SIZE_LIMIT_MB = 500       # 用于推理的权重解压后合计上限
INFERENCE_TIME_LIMIT_MS = 100   # 单张端到端上限
TEAM_NAME = os.environ.get("CXR_TEAM", "队伍名")

# 预训练来源登记表：申报 A1~A6 直接从这里取
# 注意：url 必须是公网可直接访问、无需登录的直链；license 必须是允许赛事使用的开源许可证
PRETRAIN_SOURCES = [
    {
        "name": "torchvision ConvNeXt-Base（ImageNet-1k 预训练）",
        "url": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        "license": "BSD-3-Clause（torchvision）",
        "upstream_data": "ImageNet-1k（通用自然图像，非医学）",
        "model_card": "https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html",
    },
]
