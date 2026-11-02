# 胸部 X 光多标签分类 · 合规版流水线（baseline1）

> **一句话**：这是 `baseline/` 的**合规重写版**。原方案用的 EVA-X（MIMIC-CXR 预训练）与
> CheXFound（CheXpert 预训练）属于「以医学影像为主数据集预训练」，被 `规则.md` 第 1 条与
> `规则补充.md` 二明确禁止；本版改用 **torchvision 的 ImageNet 预训练骨干**，并利用
> 「官方训练集自监督预训练」（补充四明确允许）来补医学域适应。
>
> **不止合规，也更强**：ImageNet 骨干 + 自监督 + 更充分的微调（12+ epoch、差分学习率、
> EMA、TTA、模型集成），目标是在 500MB / 100ms 约束下做到单模型最强。

---

## 1. 与 baseline/ 的四个关键区别

| | baseline/（旧） | baseline1/（本目录） |
|---|---|---|
| **预训练权重** | EVA-X（MIMIC-CXR）、CheXFound（CheXpert）→ **违反规则第 1 条** | torchvision **ImageNet-1k**（通用自然图像）→ 合规 |
| **依赖** | timm / albumentations / opencv / fvcore / transformers | **只有 torch + torchvision**（与复赛容器一致） |
| **域适应** | 靠医学预训练权重 | 靠**官方训练集自监督预训练**（SimCLR，补充四允许） |
| **合规交付物** | 无 | **cost.py**（第 7 条）+ `check_compliance.py` + `make_package.py`（附录 A） |

保持不变的好东西：study 级聚合（多图取均值，规则补充七明确允许的「患者平滑」）、
多热标签编码、加权 Focal + BCE、EMA、TTA、checkpoint 平均、模型集成、权重扫描。

---

## 2. 合规要点（详见 `README_COMPLIANCE.md`）

| 规则 | 我们怎么做 |
|---|---|
| 第 1 条 只用非医学预训练 | 全部用 torchvision ImageNet-1k 权重；`config.PRETRAIN_SOURCES` 登记名称/直链/许可证/上游数据 |
| 第 2 条 只用官方数据训练 | 训练只用 `cxr_first/` 的 train（可选 valid）；不用任何外部医学数据 |
| 第 3 条 验证集可并入但要申报 | `--data train+valid` 前先把所有选择定死，并在清单里注明 |
| 第 4/6 条 增强与推理 | 训练增强随机；推理只保留确定性变换（TTA 6 种固定），`model.eval()` + `no_grad` |
| 第 5 条 不碰测试集 | 测试集只用于最终推理，不做任何统计量/阈值/ckpt 选择 |
| 第 7 条 100ms + cost.py | `cost.py` 复用 `inference.py`（与 `predict.py` 同一份代码），100 张×3 轮 + 一致性核验 |
| 第 8 条 ≤500MB | 默认 ConvNeXt-Base fp32 ≈ 354MB；`check_compliance.py` 自动核 |
| 补充四 允许自监督 | `ssl_pretrain.py`：仅官方 train split 做 SimCLR，须在清单申报 |

---

## 3. 快速开始（GPU 机上，最高分路线）

> 核心约束：**最终推理的权重合计 ≤500MB**，一个 convnext_base 就占 334MB → 不能用多模型集成。
> 所以提分靠：**SSL 域适应 + 多 seed 训练 + 权重平均（soup，仍是单模型体积）+ TTA**。

```bash
cd ~/cxr/baseline1
pip install -r requirements.txt          # 只需 torch/torchvision/numpy/pillow/tqdm

# ① 预处理：直接复用 baseline/ 已做好的 data600（33984 张 600² 小图），无需重跑
ls ../baseline/data600/train | wc -l     # 确认存在；没有就 python preprocess.py

# ② 自监督预训练（1.5~2 小时，规则补充四允许）
python ssl_pretrain.py --epochs 20

# ③ 三个 seed 各训一次（每个约 60~80 分钟）
for s in 48 49 50; do python train.py --ssl runs/ssl/ssl_backbone.pth --epochs 15 --seed $s --out runs/s$s; done

# ④ 权重平均 → 仍是一个 334MB 的模型
python soup.py --ckpt runs/s48/ema_f15.pth runs/s49/ema_f15.pth runs/s50/ema_f15.pth \
    --out runs/soup.pth --eval-split valid

# ⑤ （推荐）全量重训：配方定死后用 train+valid 再训 3 个 seed，同样 soup
for s in 48 49 50; do python train.py --ssl runs/ssl/ssl_backbone.pth --epochs 15 --seed $s \
    --data train+valid --out runs/final_s$s; done
python soup.py --ckpt runs/final_s48/ema_f15.pth runs/final_s49/ema_f15.pth runs/final_s50/ema_f15.pth \
    --out runs/final_soup.pth

# ⑥ 推理（TTA）+ 出提交
CKPT=runs/final_soup.pth
python predict.py --split valid --ckpt $CKPT --tta --out runs/pred_valid.npz
python predict.py --split test  --ckpt $CKPT --tta --out runs/pred_test.npz
python ensemble.py --split valid --inputs runs/pred_valid.npz --out /tmp/val.csv
python ensemble.py --split test  --inputs runs/pred_test.npz  --out submission.csv

# ⑦ 合规自检 + 耗时核验 + 打包
python check_compliance.py --ckpt $CKPT
python cost.py submission.csv pictures_100 pictures_5 --meta runs/pred_test_meta.json
python make_package.py --ckpt $CKPT --submission submission.csv --team 你的队名
```

**时间紧的最短路径**：② 跳过 → ③ 只训 1 个 seed → ⑥ → ⑦，约 2 小时先拿一个合规基线分。
每步的预期输出、决策规则（soup 与单模型怎么选）见 **`README_TRAIN.md`**。

---

## 4. 文件功能

### 训练与推理
| 文件 | 职责 |
|---|---|
| `config.py` | 路径/骨干/超参/提交口径/合规参数，全部集中在这里 |
| `common.py` | CSV 读取、`7;9`→10 维多热、numpy 版 Macro AUC、study 聚合、写提交+自检 |
| `dataset.py` | Dataset（三件套返回约定）+ 纯 torchvision 的训练/验证/TTA/SSL 变换 |
| `models.py` | torchvision 骨干（convnext/efficientnet/swin/vit/resnet）+ 10 类头；参数组；SSL 权重加载 |
| `losses.py` | Focal + 类别加权 BCE（effective-number pos_weight） |
| `ema.py` | EMA 权重平均 |
| `preprocess.py` | 46GB 大图 → 600² 小图（与 baseline 共用，已存在则跳过） |
| `ssl_pretrain.py` | **SimCLR 自监督预训练**（仅官方 train split，补充四允许） |
| `train.py` | 有监督微调主循环（AMP、DataParallel、差分学习率、cosine、EMA、逐轮存 ckpt；支持 `--train-csv/--val-csv` 跑 K 折） |
| **`soup.py`** | **权重平均（model soup）**：多个 seed 合成一个模型，体积不变 → 500MB 约束下的提分手段 |
| **`make_folds.py`** | **患者级 K 折切分**（按 Subject_id 分组，杜绝跨折泄漏），产出 fold_*_train/val.csv |
| `inference.py` | **推理核心**：`load_model_for_inference` + `predict_paths`（predict.py 与 cost.py 共用） |
| `predict.py` | 批量推理 → npz + `*_meta.json`（记录 tta/ckpt/尺寸，供 cost.py 对齐） |
| `ensemble.py` | 多 npz 加权平均 → study 聚合 → `submission.csv` + 自检 |
| `scan_weights.py` | 在 valid 上扫两个模型的最优集成权重 |

### 合规交付物
| 文件 | 对应规则 |
|---|---|
| `cost.py` | 第 7 条：100 张 ×3 轮计时 + 与 submission.csv 的一致性核验 |
| `check_compliance.py` | 第 1/6/7/8 条：体积、哈希、许可证、推理确定性、A1~A6 申报表 |
| `make_package.py` | 附录 A：生成 `队伍名_最终提交材料.zip`（code/model/pretrain_model/licenses/cost.py/清单.xlsx） |
| `licenses/` | 许可证全文（BSD-3-Clause） |
| `README_COMPLIANCE.md` | 逐条规则对照 + 申报填写指引 |

---

## 5. 数据流（最高分路线）

```
cxr_first/{train,valid,test}.csv  +  原图
        │  （复用 baseline/data600，无需重跑）
        ▼
   data600/ 600² 小图
        │
        ├──► ssl_pretrain.py ──► runs/ssl/ssl_backbone.pth   （仅官方 train，补充四允许）
        │                              │
        ▼                              ▼
   train.py --ssl --seed 48/49/50 ──► runs/s48|s49|s50/ema_f15.pth   （三个 seed）
        │                              │
        │                       soup.py（权重平均，仍是 1 个 334MB 模型）
        │                              ▼
        │                        runs/soup.pth → valid 上对比 → 定配方
        │                              │
        │                    train.py --data train+valid --seed 48/49/50
        │                              ▼
        │                        runs/final_soup.pth   （全量数据，最终模型）
        ▼                              │
                    predict.py（TTA）──► pred_{valid,test}.npz + *_meta.json
                              │
                    ensemble.py ──► submission.csv（14140 行）
                              │
        check_compliance.py ──► cost.py ──► make_package.py ──► 队伍名_最终提交材料.zip
```

> 需要系统性调参时，用 `make_folds.py` 生成患者级 5 折（每折验证 2211 个患者，
> 比官方 valid 的 228 个可靠得多），逐折训练 `train.py --train-csv --val-csv` 比较配方；
> 但 K 折的 5 个模型**不能同时提交**（超 500MB），最终仍回到全量重训。

---

## 6. 常见问题

**Q：`data600` 要重新预处理吗？**
不用。本目录默认指向 `../baseline/data600`（33984 张 600² 小图），600 比所有候选骨干的
输入尺寸都大，从这里再 resize 即可。想独立一份就设 `CXR_PREP_ROOT` 并跑 `preprocess.py`。

**Q：ImageNet 权重下载失败？**
首次运行会从 `download.pytorch.org` 下载（convnext_base 约 350MB）。离线环境可在有网机器上
下载后放入 torch hub 缓存（`~/.cache/torch/hub/checkpoints/`），文件名需与官方一致。

**Q：为什么默认 ConvNeXt-Base？**
500MB 上限内它是最强的单模型之一（88.6M 参数，fp32 约 354MB），且是 torchvision 原生实现、
BSD 许可证。想更快可换 `convnext_small`（约 200MB），想集成可 `convnext_small + efficientnet_v2_m`
（合计约 418MB，仍在限内）。

**Q：能不能把两个模型集成后提交？**
可以（规则明确允许多模型集成），但**推理时加载的权重总和**要 ≤500MB。两个 base 级模型会超，
必须用小模型组合。

**Q：`cost.py` 的一致性核验为什么可能报不一致？**
它把 100 张按 study 聚合后与 submission 比对。若某个 study 的图没有全部出现在这 100 张里，
聚合结果无法精确重建——脚本会单独标注这类情况，不计入不一致。正式提交前建议确认主办方
给的 100 张是否覆盖完整 study。
