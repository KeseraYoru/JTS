# 胸部 X 光多标签分类 · 基线代码（由 Kaggle 亚军方案适配本比赛数据）

本目录是从 Kaggle **Grand X-Ray Slam**（Div A 亚军 / Div B 季军）方案移植适配的
完整训练-推理-提交流水线。原方案是 Jupyter notebook，这里改成了模块化脚本，
并针对本比赛的数据集结构做了四处关键适配（见下文「与 Kaggle 原版的差异」）。

原仓库：`../Grand-X-Ray-Slam-competition/`（含原始 notebook 与 documentation.pdf）。

> 📌 **按你的场景选入口**：
> - **要跑完整实验**（训两个模型 → 集成 → 提交 → 下载回本地）→
>   [`README_TRAIN.md`](README_TRAIN.md)（线性操作清单，照着跑即可）
> - **想搞懂「代码为什么这样写」/ 查每个文件干什么** →
>   [`README_CHANGES.md`](README_CHANGES.md)（改动注释集 + 文件功能 + 踩坑实录）
> - **想系统学一遍多标签分类怎么做的** →
>   [`TUTORIAL_多标签分类.md`](TUTORIAL_多标签分类.md)（从概念到搭代码，教学向）
> - 在 **MacBook** 上（预处理/冒烟验证）→ [`README_MAC.md`](README_MAC.md)
> - **首次上 GPU 机**（上传数据/装环境，一次性准备）→ [`README_GPU.md`](README_GPU.md)
> - 本文是总览，先读「与 Kaggle 原版的差异」表

---

## 1. 环境安装

```bash
cd baseline
pip install -r requirements.txt
```

- 训练机（学校机房 GPU）装上面全部即可；
- 只跑 **EVA-X** 一个模型的话，`fvcore` / `open_clip_torch` 不用装；
- 要跑 **CheXFound**，按 `README_TRAIN.md` 第 2.1 节准备第三方代码和权重
  （Google Drive 国内只能本地下载后人工传输），就绪标志是
  `python3 check_chexfound.py` 全部通过。

## 2. 数据准备（预处理 46GB 大图）

原始图 1409~4280px、平均 1.5MB，直接训练 I/O 太慢。先统一缩放到 600×600
（Kaggle 方案同款思路），目录层级保持不变：

```bash
# 在任意有 CPU 的机器上跑都行（约 30 分钟，8 核）
python preprocess.py                # train + valid + test 全部
python preprocess.py --split train  # 也可只处理某一个 split
python preprocess.py --limit 100    # 调试：只处理前 100 张
```

输出到 `data600/{train,valid,test}/subject_x/study_y/pic_z.jpg`（约 3.5GB）。
**把这个目录拷贝到训练机**，并在训练机上把 `config.py` 的 `PREP_ROOT`
（或环境变量 `CXR_PREP_ROOT`）指向它。

> 如果机房磁盘紧张，也可以跳过预处理、把 `PREP_ROOT` 直接指向 `../cxr_first`
> 用原图训练——能跑但每个 epoch 慢 5~10 倍，不推荐。

## 3. 训练

```bash
# EVA-X（主模型，448×448，T4 16GB 下 batch 16，6 epoch 约 5~6 小时）
python train.py --model evax

# CheXFound（第二模型，512×512，冻结 backbone 只训分类头，约 2~3 小时）
python train.py --model chexfound

# 常用可选项
python train.py --model evax --epochs 3 --batch-size 8   # 先小跑验证流程
python train.py --model evax --data train+valid          # 联合 train+valid 出正式模型
python train.py --model evax --resume runs/xxx/model_f3.pth
```

每个 epoch 在 `runs/<模型>_<时间戳>/` 下保存：

| 文件 | 说明 |
|---|---|
| `model_f{N}.pth` | 第 N 个 epoch 的普通权重 |
| `ema_f{N}.pth` | 第 N 个 epoch 的 EMA 权重（**推理默认用这个**） |
| `best.pth` | 验证集 study 级 Macro AUC 最好的 EMA 权重 |
| `metrics.json` | 每个 epoch 的 loss / 图像级 AUC / study 级 AUC（实验台账用） |

训练时会同时打印**图像级**和 **study 级**两组 Macro AUC 和 10 个类各自的 AUC。
比赛按 `Study_id` 评分，所以 **study 级那组才是选型依据**；每类 AUC 用来定位
薄弱类别（十类等权，罕见类排错一样扣分）。

## 4. 推理（TTA）

```bash
# EVA-X：第 5、6 epoch 的权重 + TTA（Kaggle 方案配置）
python predict.py --model evax --split test --tta \
    --ckpt runs/evax_xxx/ema_f5.pth runs/evax_xxx/ema_f6.pth \
    --out runs/evax_xxx/pred_test.npz

# CheXFound：第 5、6 epoch 权重，不用 TTA
python predict.py --model chexfound --split test \
    --ckpt runs/chexfound_xxx/ema_f5.pth runs/chexfound_xxx/ema_f6.pth \
    --out runs/chexfound_xxx/pred_test.npz

# 在 valid 上推理会顺带打印 study 级分数，用于模型选型
python predict.py --model evax --split valid --tta --ckpt ... --out runs/evax_xxx/pred_valid.npz
```

多个 `--ckpt` 会像 Kaggle 一样对**预测概率**取平均（不是平均权重）。

## 5. 生成提交文件

```bash
# 先自评（valid 集上看集成效果）
python ensemble.py --split valid \
    --inputs runs/evax_xxx/pred_valid.npz runs/chexfound_xxx/pred_valid.npz \
    --out val_check.csv

# 正式提交：EVA-X + CheXFound 平均（Kaggle 最终方案）
python ensemble.py --split test \
    --inputs runs/evax_xxx/pred_test.npz runs/chexfound_xxx/pred_test.npz \
    --out submission.csv
```

`submission.csv` 为长表：`Subject_id,Study_id,Predict_class,Probability`，
默认每个 study 输出全部 10 类概率（1414×10 = 14140 行），概率保留 6 位小数。
生成后会自动做自检（study 覆盖、行数、类别 id、概率范围）。

> ⚠️ **提交口径待向主办方确认**：题面写「默认 0.5 阈值、可多行输出」，但 Macro AUC
> 需要完整概率矩阵。本代码默认交全 10 类；若主办方明确要求只交 >0.5 的类，
> 把 `config.py` 的 `SUBMIT_ALL_CLASSES` 改成 `False` 再重新生成即可。
> 每日只有 2 次提交机会，不要拿名额试格式。

## 6. 与 Kaggle 原版的差异（适配点）

| Kaggle 原版 | 本比赛数据/约束 | 本代码的改法 |
|---|---|---|
| 扁平图像目录 + `Image_name` 列 | `subject/study/pic` 三级目录 | `common.py` 用 CSV 三列拼路径 |
| 14 个 0/1 标签列 | 1 列 `7;9` 分号分隔 | `parse_predict_class` → 10 维多热 |
| 14 类（含 Fracture 等） | 10 类 | `config.CLASS_IDS` 固定 1~10 |
| 图像级宽表提交 | **study 级**长表提交 | `aggregate_by_study` 多图 mean 聚合 |
| notebook 单文件 | 模块化脚本 | 8 个 .py 文件，便于三人分工 |
| Kaggle GPU（可开大 batch） | T4 16GB | batch 16（EVA-X）/ 32（CheXFound） |

保持不变的核心手法：600² 预缩放、WeightedFocalBCE（β=0.9999）、EMA(0.9997)、
layer-wise 学习率、TTA 6 变换、epoch 5+6 预测平均、双模型集成。

## 7. 注意事项

1. **复赛约束**：模型压缩前 ≤500MB、单张 ≤100ms。EVA-X base（fp32 约 344MB）
   单独提交是满足的；EVA-X+CheXFound 集成版只用于初赛刷分，复赛交单个
   EVA-X（或后续蒸馏小模型）。TTA 会把推理耗时乘 6，复赛需评估。
2. **EVA-X 权重下载**：首次训练自动从 HuggingFace 下载（约 344MB）。
   国内打不开时设 `CXR_HF_ENDPOINT=https://hf-mirror.com`，或手动下载后放到
   `weights/eva_x_base_patch16_merged520k_mim.pt`。
3. **显存不够（OOM）**：`--batch-size 8`；再不够就把 `config.EVAX_IMG_SIZE`
   降到 320（需重新预处理到 600² 不受影响，只是模型输入变小）。
4. **复赛容器离线**：最终模型要能离线推理，权重需随镜像打包；
   容器只保证 PyTorch+torchvision，所以复赛的推理脚本要脱离 timm/albumentations
   （本仓库的 `predict.py` 依赖它们，复赛前需改造，留待后期）。
5. **实验台账**：每次实验的 `runs/` 目录名、metrics.json、提交分数请记录到
   团队共享表格（日期 / 改动 / 每类 AUC / 提交分）。
6. **git**：建议在 `计挑赛/` 根目录 `git init`，本目录的 `.gitignore` 已排除
   大文件（data600/runs/weights/third_party 都不会被提交）。

## 8. 目录结构

```
baseline/
├── README.md              本文（总览）
├── README_TRAIN.md        训练流程手册（双模型→集成→提交→下载）
├── README_MAC.md          Mac 使用指南
├── README_GPU.md          GPU 机上机指南（上传/环境/排查）
├── README_CHANGES.md      改动注释集：每个坑的「症状→根因→修法」+ 文件功能
├── TUTORIAL_多标签分类.md  教学：从概念到搭一套多标签分类流水线
├── requirements.txt       训练环境依赖
├── setup_chexfound.sh     准备 CheXFound 第三方代码（权重人工放）
├── check_chexfound.py     CheXFound 环境一键检查
├── config.py              所有路径与超参（调实验只改这里）
├── common.py              CSV 读取/标签编码/AUC/study 聚合/提交生成
├── dataset.py             Dataset + 数据增强（albumentations）
├── losses.py              Focal + 加权 BCE
├── ema.py                 EMA 权重平均
├── preprocess.py          46GB 大图 → 600² 小图
├── train.py               训练主循环
├── predict.py             TTA 推理 → npz
├── ensemble.py            多模型平均 → submission.csv
├── scan_weights.py        在 valid 上扫两个模型的最优集成权重
└── models/
    ├── __init__.py        模型/优化器工厂
    ├── evax.py            EVA-X（timm，含权重下载）
    └── chexfound.py       CheXFound（GLoRIA 头 40→10 类改造）
```
