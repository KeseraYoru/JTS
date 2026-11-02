# -*- coding: utf-8 -*-
"""全局配置：所有路径、超参、开关都集中在这一个文件里。

调实验只改这里，或者用环境变量覆盖（多人协作时互不影响）。
所有脚本都从 baseline/ 目录下运行：python train.py --model evax
"""
import os

# ---------------- 路径 ----------------
# 原始数据根目录（46GB 大图 + train/valid/test.csv）
DATA_ROOT = os.environ.get("CXR_DATA_ROOT", "../cxr_first")
# 预处理输出根目录（600x600 小图，约 3.5GB，训练/推理都读这里）
PREP_ROOT = os.environ.get("CXR_PREP_ROOT", "./data600")
# 训练输出目录（ckpt、指标日志）
RUNS_DIR = os.environ.get("CXR_RUNS_DIR", "./runs")
# 预训练权重目录（EVA-X 的 .pt 会自动下载到这里）
WEIGHTS_DIR = os.environ.get("CXR_WEIGHTS_DIR", "./weights")
# CheXFound 第三方代码与权重（需手动准备，见 README 第 2 节）
CHEXFOUND_REPO = os.environ.get("CXR_CHEXFOUND_REPO", "./third_party/CheXFound")
CHEXFOUND_MODEL_DIR = os.environ.get(
    "CXR_CHEXFOUND_MODEL_DIR", "./third_party/CheXFound_Model"
)

SPLITS = ("train", "valid", "test")

# ---------------- 类别 ----------------
# 顺序即多热编码顺序，也与提交文件里的 Predict_class 一一对应
CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CLASS_NAMES_ZH = {
    1: "纵隔心影增大", 2: "气胸", 3: "肺部实变", 4: "肺炎", 5: "肺水肿",
    6: "心影扩大", 7: "肺不张", 8: "肺部阴影", 9: "胸腔积液", 10: "无异常发现",
}
NUM_CLASSES = len(CLASS_IDS)

# ---------------- 预处理 ----------------
PREP_SIZE = 600        # 预处理统一缩放到 600x600（与 Kaggle 方案一致）
PREP_QUALITY = 95      # JPEG 质量

# ---------------- 训练通用 ----------------
MODEL = "evax"         # evax | chexfound
EPOCHS = 6             # Kaggle 方案：EVA-X 6 epoch，CheXFound 3+3=6 epoch
BATCH_SIZE = 32        # 服务器 4x RTX 5090 32GB 的安全值（T4 16GB 时代是 16）；
                       # 显存充裕想更快可加到 64；单卡跑 OOM 就降回 16
NUM_WORKERS = 8        # DataLoader 进程数，按 CPU 核数调
USE_AMP = True         # 混合精度训练（省显存、提速，必须开）
EMA_DECAY = 0.9997     # 指数移动平均衰减（Kaggle 同值）
WARMUP_RATIO = 0.05    # cosine 学习率预热的比例
WEIGHT_DECAY = 1e-4
SEED = 48              # Kaggle 同值

# EVA-X（输入 448x448，从 HuggingFace 下载 MIMIC 预训练权重）
EVAX_IMG_SIZE = 448
EVAX_WEIGHT_URL = (
    "https://huggingface.co/MapleF/eva_x/resolve/main/"
    "eva_x_base_patch16_merged520k_mim.pt"
)
EVAX_WEIGHT_FILE = "eva_x_base_patch16_merged520k_mim.pt"
# 国内网络打不开 huggingface.co 时，可设 CXR_HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT = os.environ.get("CXR_HF_ENDPOINT", "https://huggingface.co")

# CheXFound（输入 512x512，冻结 backbone 只训 GLoRIA 头）
CHEXFOUND_IMG_SIZE = 512
CHEXFOUND_LR = 1e-4    # 只训分类头，学习率比 EVA-X 大
CHEXFOUND_BATCH_SIZE = 32
CHEXFOUND_ORIG_NUM_CLASSES = 40   # 官方 glori.pth 是 40 类（CheXpert 竞赛），要改造成 10 类

# ---------------- 损失（Focal + BCE，专治类别不均衡） ----------------
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
FOCAL_BCE_MIX = 0.7    # 最终损失 = 0.7 * focal + 0.3 * 加权 BCE
POS_WEIGHT_BETA = 0.9999   # effective number of samples 的 beta
POS_WEIGHT_CAP = 10.0      # 正样本权重上限，防止极端值

# ---------------- 推理 / 提交 ----------------
STUDY_AGG = "mean"     # study 级聚合方式：mean（多图概率平均）| max
SUBMIT_ALL_CLASSES = True   # True: 每个 study 交全 10 类概率（AUC 需要完整矩阵）
SUBMIT_THRESHOLD = 0.5      # 仅当 SUBMIT_ALL_CLASSES=False 时生效
PROB_DECIMALS = 6           # 概率保留 6 位小数，避免并列拉低 AUC
