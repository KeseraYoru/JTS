# README_CHANGES.md —— 这个代码库是怎么长出来的

> **这份文档是干什么的**：它不是操作手册（操作看 `README_TRAIN.md`），而是一份
> **「为什么每行关键代码长这样」的注释集**。把我们从拿到数据到第一次提交踩过的
> 每一个坑、每一个改动决定都记下来，方便：
> - 你（或队友）改代码前知道这段为什么这么写，别改回去；
> - 复赛写文档、答辩讲思路时有据可查；
> - 下次做别的任务时，这套「怎么和一个不认识的第三方库/多卡环境打交道」的经验能直接复用。
>
> 阅读顺序建议：先看「第一部分：改动时间线」了解全貌，再看「第二部分：文件功能」。

---

## 第一部分：改动时间线（症状 → 根因 → 修法 → 验证）

改动分六类。**每条都按「当时看到什么 → 真正的原因 → 怎么改 → 怎么确认改对了」记录。**

---

### A. 数据结构适配（一开始就必须做的 4 处结构性改动）

Kaggle 原方案（Grand-X-Ray-Slam，Division A 亚军）是为另一个数据集写的，拿到我们这份
数据后有 4 处**必须**改，否则连跑都跑不起来：

| # | Kaggle 原版 | 我们的数据 | 改法 | 落在哪个文件 |
|---|---|---|---|---|
| A1 | 扁平目录 + `Image_name` 列 | `subject_x/study_y/pic_z.jpg` 三级目录 | 用 CSV 的 `Subject_id`+`Study_id`+`Dicom_id` 三列拼路径 | `common.py` |
| A2 | 14 个 0/1 标签列 | 单列 `Predict_class`，内容是 `7;9` 这种分号串 | 解析成 10 维 0/1 多热向量 | `common.py: parse_predict_class / encode_labels` |
| A3 | 图像级宽表提交 | **study 级长表**提交 | 新增 study 聚合（多图 mean）→ 长表 | `common.py: aggregate_by_study / write_submission` |
| A4 | 14 类 | 10 类 | CheXFound 头 40→10，EVA-X 头直接建 10 类 | `models/chexfound.py: _resize_glori_classes` |

**数据勘察时发现的 5 个事实**（决定了后面所有设计）：
1. 标签是 **study 级**：18513 个多图 study 里，标签不一致的数量为 **0** → 同一 study 的所有图共享标签，可以按「图」训练、按「study」评分。
2. 类别 10（无异常）与其余 9 类**互斥**（含 10 的行 = 纯 10 的行 = 12649）→ 多热编码天然体现，不需要额外约束。
3. 类别**极度不均衡**：最罕见类 1 只有 894 个正样本（2.95%），类 10 占 41.69%（相差 14 倍）。
4. 图像是 8bit 灰度大图（1409~4280 px，train 46GB），**无固定尺寸** → resize 方案必须在训练/推理两端写死一致。
5. 三个 split **无患者/检查重叠**（subject 与 study 都无交叉），很干净。

---

### B. macOS 本地验证暴露的坑（这些坑在 Linux 上有一个根本不会暴露）

#### B1. TTA 亮度变换无法被 pickle → macOS 上推理直接崩
- **症状**：Mac 上跑 `predict.py --tta`，前 4 种 TTA 正常，第 5 种（亮度 0.95）时崩：
  `AttributeError: Can't get local object 'make_tta_tfms.<locals>.scale_brightness.<locals>.apply_fn'`
- **根因**：macOS 的 DataLoader worker 用 **spawn** 启动，要把 dataset 连 transform 一起 pickle 传给子进程；而亮度变换是个**嵌套函数**（局部函数），pickle 不了。Linux 默认 fork 不经过 pickle，所以**这个 bug 在 GPU 机上永远不会出现**——是 Mac 冒烟测试把它逼出来的。
- **修法**：实现提到模块级 `_scale_brightness_impl`，用 `functools.partial` 包装（partial 可 pickle，闭包不行）。
- **验证**：6 种 TTA 逐个 `pickle.dumps` 全部通过。
- **文件**：`dataset.py`

#### B2. Dataset 返回结构可变 → valid 推理把「标签」当成了「图片 id」
- **症状**：npz 里的 `dicom_ids` 变成了 `(8, 10)` 的 float 数组（不是字符串 id）。
- **根因**：`__getitem__` 在 `with_labels=True` 时第二个返回值是**标签张量**而不是 `Dicom_id`。于是 predict 在 **valid** split 上把标签当成 id 存了进去（开头全是 0，正好是「无异常」样本的标签）。test split 不触发，**valid 推理（选型依据）和集成对齐全崩**。
- **修法**：Dataset 统一返回 `(img, dicom_id, labels)` 三件套，labels 在推理时用零向量占位；train/predict 共 4 处循环解包同步改；ensemble 的 id 比较加 `str()` 兜底。
- **验证**：重跑后 npz 的 `dicom_ids` dtype 恢复 `<U10`（字符串），ensemble 自检通过。
- **文件**：`dataset.py`、`train.py`、`predict.py`、`ensemble.py`

> 💡 这是本项目最重要的一条**设计约定**：`Dataset.__getitem__` 永远返回三件套，别再改回两件套。

---

### C. 4 卡服务器（4×RTX 5090）暴露的坑

#### C1. DataParallel 包装后优化器构建崩溃
- **症状**：服务器 4 张卡，train.py 检测到多卡自动包 `DataParallel`，随后构建 EVA-X 优化器时崩（访问 `model.backbone` 失败）。
- **根因**：`DataParallel` 包装后，原模型的属性访问不到（要 `model.module.xxx`）。Mac 单卡永远测不到这条路径。
- **修法**：构建优化器前先剥包装 `model = getattr(model, "module", model)`（参数是同一批张量，优化器照常生效）。
- **验证**：在 Mac 上用**真 EVA-X 模型 + DataParallel 包装**实测：9 个参数组、85.7M 参数全覆盖、forward+backward+step 全通过。
- **文件**：`models/__init__.py`、`models/evax.py`

#### C2. CheXFound：第三方库强加的 DDP 包装 → 第 2 个 step 必崩（最难的一个）
- **症状**：`train.py --model chexfound` 第 1 个 step 正常（loss 0.5801 打出来了），**第 2 个 step 的 forward** 崩：
  ```
  RuntimeError: Expected to have finished reduction in the prior iteration ...
  Parameter indices which did not receive grad for rank 0: 1
  ```
- **根因链**（读 chexfound 源码才看得懂，三层）：
  1. 我们调 `setup_and_build_model` → 库的 `utils/config.py: default_setup` **无条件** `init_process_group(NCCL)`；
  2. `eval/classification/utils.py: setup_glori` 发现进程组已启用，就把 glori 头包进 **DDP**（`find_unused_parameters` 用默认 `False`）；
  3. 我们的 `_resize_glori_classes` 在 **DDP 包装之后**用 `setattr` 把 40 类的 `duplicate_pooling_bias` 换成 10 类新 Parameter——而 DDP 记录的还是**旧参数对象**，它已脱离模块、永远收不到梯度（该参数正好是 DDP 参数列表的 **index 1**，与报错完全对应）。
- **修法（两步，根因消除 + 双保险）**：
  1. `_suppress_library_distributed()`：在任何 chexfound 导入之前，把库的 `distributed.enable` 打成 no-op → 没有进程组 → `is_enabled()` 为 False → **库根本不会包装 DDP**；
  2. 保留 `isinstance(glori, DDP)` 拆包 + `destroy_process_group()` 作为双保险（正常流程下是空操作）。
- **验证**：
  - 用**真实的** `chexfound.distributed` 模块本地模拟：补丁后 `default_setup` 的调用序列安全（`get_global_rank()` 返回 0 不崩），`setup_glori` 判断不包装；
  - 用 Gloo/CPU 双进程**完整复现**了服务器报错（连「第 1 个 iteration 正常、第 2 个崩」的细节都对上），修复路径两个 iteration 全过、新 bias 有梯度、优化器能看到 10 维新参数。
- **文件**：`models/chexfound.py`
- **教训**：第三方科研库的 setup 函数常夹带私货（初始化进程组 / 包装 DDP）。**拿到返回的模型先检查有没有被包壳；在包装之后做参数手术（setattr）必崩 DDP。**

#### C3. 「修复了怎么还崩」——服务器跑的是旧代码
- **症状**：修完 C2 后用户跑冒烟，报错**一字不差**。
- **根因**：诊断依据是——修复若生效，DDP 会被整个拆掉，日志里不可能再出现 `[rank0]:` 开头的 DDP traceback。所以**服务器上的代码没更新**（rsync 那步被跳过）。
- **修法**：流程上改成「先同步 + 验证版本，再跑」。给出自查命令：
  `grep -c _suppress_library_distributed models/chexfound.py` 应输出 **2**。
- **文件**：`README_TRAIN.md` 附录 B 已加该自查项。

---

### D. 第三方库版本坑

#### D1. albumentations 2.x 把 CoarseDropout 的参数**静默忽略**
- **症状**：警告 `Argument(s) 'max_holes, max_height, max_width' are not valid for transform CoarseDropout`。
- **根因**：2.x 里这三个参数改名了（改成了 `num_holes_range` / `hole_*_range` 比例制），**旧参数不报错、被忽略**，增强一直在用默认值跑，偏离 Kaggle 配方（20×20 像素单孔）。
- **修法**：新增 `_coarse_dropout()`，按 albumentations 大版本号选参数。
- **验证**：448 和 512 输入下黑洞都精确回到 **20×20 = 400 像素**，警告消失，完整 pipeline 输出 `(3, 448, 448) float32`。
- **文件**：`dataset.py`

#### D2. 两组「pip 包名 ≠ 导入名」
| pip 装的名字 | import 的名字 | 踩坑表现 |
|---|---|---|
| `open_clip_torch` | `open_clip` | pip 明明装成功（3.3.0），检查脚本却报缺失 |
| `scikit-multilearn` | `skmultilearn` | 镜像和 PyPI 都找不到 `skmultilearn`，pip 报「无匹配发行版」 |

- **文件**：`check_chexfound.py`（依赖表改成 `(导入名, pip 包名)` 二元组）、三份文档的安装命令。

#### D3. CheXFound 的真实依赖不止 2 个（最终 9 个）
- 逐个试错太慢 → 对 clone 下来的源码做**全包静态 import 分析**，一次数完训练链上真正用到的：
  `fvcore / open_clip_torch / omegaconf / scikit-multilearn / torchmetrics / iopath / pandas / transformers / scikit-image`
- **明确不装**：`xformers`（官方 pin 的 0.0.18 与 torch 2.14+CUDA 13 不兼容，代码自动回退原生 attention，只有 warning）、`mmcv / mmsegmentation / cuml / peft / submitit`（只在官方完整预训练脚本里用，我们不跑）。
- **文件**：`check_chexfound.py`（一键检查 4 类：代码目录 / 4 个权重文件含大小+zip 魔数校验 / 9 个 pip 包 / 6 个子模块导入）

---

### E. 工程流程坑

#### E1. gitignore 行内注释让规则失效（差点把 2.5G 图片提交进仓库）
- **症状**：`git add -An` 演练发现 `data600/` 的 2.5G 图片**会被 git 跟踪**。
- **根因**：原 `baseline/.gitignore` 写成 `data600/  # 预处理图像`——**gitignore 的 `#` 只在行首才是注释**，行内的 `#` 和后面的文字会变成文件名的一部分，导致 `data600/`、`runs/`、`weights/` 三条规则全部失效。
- **修法**：注释独立成行。
- **验证**：用 `git check-ignore` 逐条实测 4 个目录全部命中（还重建目录测了 `runs/`）。
- **文件**：`baseline/.gitignore`、根 `.gitignore`（新增，忽略 `cxr_first/`、`*.zip`、后来的 `results/`）

#### E2. rsync 的 `--partial` 与 `--exclude` 冲突（macOS rsync 2.6.9 的 bug）
- **症状**：上传时报 `link_stat ".../计挑赛/data600/." failed` 且大目录没排除（实测会连 data600+weights 一起传，共 3.6G）。
- **根因**：对照实验（A/B/C/D 四组）确认——`--partial` 与带尾斜杠的 `--exclude 'dir/'` 同用时，macOS 自带 rsync 2.6.9 会让 exclude 失效。
- **修法**：传代码的那条命令**去掉 `--partial`**（代码只有 140K，秒传）；传数据/权重的命令无 exclude，`--partial` 照用（断点续传）。
- **另**：rsync 只创建目标的**最后一级**目录，远程 `~/cxr` 父目录要先 `mkdir -p`。
- **文件**：`README_GPU.md` 2.2（三条命令改单行 + 去危险组合）、`README_MAC.md` 第 4 节同步。

#### E3. CSV 没传（图片在 data600，标注 CSV 全 Mac 独一份）
- **症状**：服务器 `FileNotFoundError: ../cxr_first/train.csv`。
- **根因**：上传计划排除了 `cxr_first/`（46G 图片），但同目录的 3 个标注 CSV 是训练/推理必需的元数据，被一起漏掉。
- **修法**：单独 rsync 3 个 CSV（2.5MB）到 `~/cxr/cxr_first/`；同时把 `read_split_csv` 的报错改成人话（直接给出该执行的命令 + 环境变量备选）。
- **文件**：`common.py`、`README_GPU.md`

#### E4. 选 checkpoint 的坑：手册写死 `ema_f5/f6`，但 EVA-X 可能是 10 epoch
- **症状**：手册 Step 1 推荐 EVA-X **10 epoch**，Step 3/附录 A 的命令却写死 `ema_f5.pth ema_f6.pth`（那是 6 epoch 的做法）→ 用在 10 epoch 的 run 上会**白丢 4 个 epoch 的提升**。
- **修法**：明确 Kaggle 配方的真实语义是「**平均每个 run 的最后两个 epoch**」，加对照表 + 先查 epoch 数的命令：
  | run 训了几轮 | 用哪两个 |
  |---|---|
  | 6 epoch | `ema_f5.pth ema_f6.pth` |
  | 10 epoch | `ema_f9.pth ema_f10.pth` |
- **文件**：`README_TRAIN.md` Step 3 + 附录 A

---

### F. 方法论修正（不是 bug，是「想错了」）

#### F1. 等权集成反而掉分 → 新增权重扫描
- **症状**：valid 上等权集成 **0.8240**，低于 CheXFound 单模型的 **0.8263**。
- **根因**：逐类对比发现，集成确实提升了 5/6/7/9/10 类，但 EVA-X 在 **1/2/3 类明显更弱**（1: 0.771→0.750，2: 0.897→0.886，3: 0.835→0.818），等权把短板硬掺进来，净效果为负。
- **修法**：新增 `scan_weights.py`——在 valid 上把权重从 0 扫到 1，直接给出最优配比（w=1.00 那行顺便就是「只用 EVA-X」的分数）。
- **验证**：用合成数据（构造一强一弱 + 重叠噪声）实测，脚本准确找到内部最优 0.25/0.75（0.9847），优于强模型单跑（0.9687）和等权（0.9755）。
- **文件**：`scan_weights.py`（新增）、`README_TRAIN.md` Step 4

#### F2. 第一次提交 0.7936 vs valid 0.8240 ——落差的三个候选解释
1. **valid 太小，估计本身就不准**（最主要）：valid 只有 859 个 study，罕见类的正样本极少——类别 1 只有 **24 个正 study**，其 AUC 标准误约 **±0.056**；十类平均后 Macro AUC 的标准误约 **±0.01**。所以 0.8240 的「真值」可能是 0.81~0.83。
   （注：已核对 valid 与 train 的类别分布几乎一致，排除了「valid 分布偏移」。）
2. **概率整体偏低，若平台按 0.5 阈值处理会显著拉低**：提交文件里各类概率均值只有 0.18~0.42，其中类别 10（真实占比 41.7%）只有 **9%** 的预测超过 0.5。AUC 是排序指标、不受整体偏移影响，但**阈值化后**就会被放大成实打实的损失。**这是最值得验证的一条**，诊断命令见本文末尾。
3. test 本身更难（无标注，无法直接验证）。

---

## 第二部分：文件功能一览

### 核心代码（15 个 .py + 1 个 .sh）

| 文件 | 职责 | 什么时候会用到 |
|---|---|---|
| `config.py` | **所有路径与超参集中一处**（模型、分辨率、batch、loss、EMA、提交口径） | 调实验只改这里 |
| `common.py` | CSV 读取、`7;9`→10 维多热、numpy 版 Macro AUC、**study 聚合**、提交文件生成+自检 | 几乎所有脚本都依赖它 |
| `dataset.py` | Dataset（**三件套返回约定**）+ 训练/验证增强 + TTA 变换组 | 训练/推理 |
| `losses.py` | Focal Loss + 按「有效样本数」加权的 BCE（`WeightedFocalBCELoss`） | 训练 |
| `ema.py` | EMA（指数移动平均）权重 | 训练 |
| `preprocess.py` | 46GB 大图 → 600² RGB 小图（多进程，支持「已存在则跳过」） | 只在 Mac 上跑一次 |
| `train.py` | 训练主循环（AMP、多卡 DataParallel、每轮存 ckpt、valid 监控） | 训练 |
| `predict.py` | 加载 ckpt →（可选 TTA）→ 输出 npz（`--ckpt` 可给多个自动平均） | 推理 |
| `ensemble.py` | 多模型预测加权平均 → study 聚合 → `submission.csv` + 自检 | 出提交 |
| `scan_weights.py` | **在 valid 上扫两个模型的最优集成权重** | 集成前 |
| `check_chexfound.py` | CheXFound 环境一键检查（代码/权重/9 个依赖/子模块导入） | 训 CheXFound 前 |
| `models/__init__.py` | 模型与优化器工厂（`build_model` / `build_optimizer`，含 DataParallel 解包） | 训练/推理 |
| `models/evax.py` | EVA-X（timm + HF 权重下载 + 10 类头 + layer-wise 学习率） | 训练/推理 |
| `models/chexfound.py` | CheXFound（ViT-Large + GLoRIA 头 40→10、冻结 backbone、**禁止库初始化 DDP**） | 训练/推理 |
| `setup_chexfound.sh` | 准备 CheXFound 第三方代码 + 校验权重就位（**不联网下载**，权重走本地传输） | 首次 |
| `requirements.txt` | 依赖清单（含 Mac 实测通过的版本组合注释） | 装环境 |

### 文档（4 份，各管一段）

| 文档 | 管什么 |
|---|---|
| `README.md` | 总览 + 与 Kaggle 原版的差异对照 + 四份文档导航 |
| `README_MAC.md` | Mac 端：预处理、传数据、CPU 冒烟验证（**Mac 不训练**） |
| `README_GPU.md` | 上机全流程：装环境 → 上传 → 训练 → 下载回 → 本地评分 → 故障排查 |
| `README_TRAIN.md` | **日常操作手册**：Step 0~6 线性流程（训两个模型 → 集成 → 出提交） |
| `README_CHANGES.md` | 本文：改动注释集 + 文件功能 |
| `TUTORIAL_多标签分类.md` | 教学：从零理解并搭一套多标签分类流水线 |

### 数据/产物目录（全部 gitignore）

```
baseline/
├── data600/      2.5G  预处理后的小图（33984 张，与 CSV 逐一对应）
├── weights/      1.1G  EVA-X 预训练权重
├── runs/               每次训练的 ckpt + metrics.json
└── third_party/  2.4G  CheXFound 代码 + 权重（本地下载后 rsync 上去）
计挑赛/results/         从服务器拉回的产物（submission.csv / npz / metrics）
```

---

## 第三部分：数据流（一句话版）

```
CSV(三列拼路径) ──► preprocess.py ──► data600/ 600² 小图
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
                 train.py --model evax            train.py --model chexfound
                        │                                   │
                   runs/evax_*/ema_f*.pth          runs/chexfound_*/ema_f*.pth
                        │                                   │
                  predict.py --tta                predict.py（无 TTA）
                        └──────────► pred_{test,valid}.npz ◄──┘
                                          │
                            scan_weights.py（定权重）
                                          │
                              ensemble.py ▼
                                   submission.csv（14140 行）
```

关键点：**训练按「图」喂，评分/提交按「study」聚合**——中间的粒度转换由
`common.aggregate_by_study` 统一负责，聚合方式（mean/max）是可调超参。

---

## 第四部分：待办 / 遗留问题

1. **验证 0.7936 的成因**（最高优先级）：在服务器上跑下面这段，看「二值化后的 AUC」
   是不是接近 0.79——如果是，说明平台按 0.5 阈值处理，需要做概率校准：
   ```bash
   python3 -c "
   import numpy as np, sys; sys.path.insert(0,'.')
   from common import read_split_csv, encode_labels, aggregate_by_study, macro_auc
   rows = read_split_csv('valid'); labels = encode_labels(rows)
   p = np.load('runs/chexfound_20261022_120718/pred_valid.npz', allow_pickle=True)['probs']
   _, sp,  _  = aggregate_by_study(rows, p);            _, _, sl = aggregate_by_study(rows, p, labels=labels)
   b = (p >= 0.5).astype(p.dtype)
   _, sp2, _  = aggregate_by_study(rows, b)
   print('原始概率 AUC :', round(float(macro_auc(sl, sp )), 4))
   print('0.5 二值化后 :', round(float(macro_auc(sl, sp2)), 4))
   "
   ```
2. **提交口径**：题面写「默认以 0.5 为阈值」，但我们按「全 10 类概率」提交（14140 行）。
   确认后决定要不要改（若平台确实只在乎阈值以上的类，就要做校准/阈值调优）。
3. **`--data train+valid`**：配方定稿后，用全部数据出正式模型（多吃 859 个检查的数据）。
4. **类别 8（肺部阴影）** 是两个模型的共同弱项（约 0.74），属于「类间相似性」这一赛题点名的难点，值得专项分析。
5. **复赛约束**：模型 ≤500MB、单张 ≤100ms、容器离线（无 timm/albumentations）——
   当前双模型集成只适用于初赛刷分，复赛需交单个模型并改造推理脚本。
