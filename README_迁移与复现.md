# 迁移与复现 · 入口文档

> 最后更新：2026-11-06　｜　清理后总占用 **55G**，但**真正需要搬的只有 6.8G**
> 这份文档是迁移时的唯一入口：先读第 1、2 节搬东西，再按第 4 节校验，第 5 节跑训练。

---

## 0. 一句话现状

合规最高分 **77.67**（双架构集成，valid 0.8027），复赛线 77.7，差 0.03。
三条有效手段已吃满（SSL +0.0258 / 跨架构集成 +0.0030 / 全量数据 +0.0047），
其余 12 条路已用实验判负。**当前最大风险不是分数，是双模型在官方 T4 上能否跑进 100ms。**

---

## 1. 清理后还剩什么

| 路径 | 体积 | 是什么 | 迁移? |
|---|---|---|---|
| `cxr_first/train/` | 46G | 30344 张训练原图 | **可选**（只有重做预处理/换分辨率源时才要） |
| `cxr_first/test/` | 3.4G | 2218 张测试原图 | **必须**（`cost.py` 核验要用真原图） |
| `cxr_first/valid/` | 2.1G | 1422 张验证原图 | **推荐**（重做 valid 预处理时用） |
| `cxr_first/*.csv` | 1.9M | 三个标注 CSV | **必须** |
| `baseline/data600/` | 2.5G | 33984 张 600² 预处理图 | **必须**（训练直接吃它） |
| `baseline1/` | 268K | 产出 77.67 的主线代码（16 个 py + 3 份 README） | **必须** |
| `baseline3/` | 156K | 新框架：换起点（IN-22k / DINOv2）+ 修掉 baseline1 的 8 个坑 | **必须**（继续训练用） |
| `results/weights_7767/` | 551M | 77.67 的两个模型权重（fp32） | **必须**（保底提交物） |
| `results/runs/` | 5.9M | 各实验的 metrics / npz 证据 | 推荐 |
| `results/submission_7767_ens.csv` | 526K | 77.67 那份提交文件 | **必须** |
| `dl/convnext_base_22k_1k_224.pth` | 338M | ImageNet-22k 起点权重（未验证的那条线） | **必须**（否则要重下 354MB） |
| `tools/screen_solution.py` | 28K | 合规红线扫描脚本 | 推荐 |
| `*.md` | ~100K | 见第 6 节文档地图 | **必须** |
| `archive/` | 80K | 历史文档（两周计划、执行手册、老师简报） | 可选 |

---

## 2. 迁移清单：三档，按你的带宽选

### 档位 A · 最小可用（6.8G）— 只继续训练 + 能出一份可核验的提交

```bash
cd /Users/mac/WorkBuddy/计挑赛
rsync -av --progress \
  baseline baseline1 baseline3 results tools dl \
  archive *.md pack_t4.sh \
  user@新设备:~/计挑赛/

# 只搬 test 原图 + CSV（train/valid 原图不搬）
rsync -av --progress cxr_first/*.csv user@新设备:~/计挑赛/cxr_first/
rsync -av --progress cxr_first/test user@新设备:~/计挑赛/cxr_first/
```

> 这个档位能：训练（data600 已含 train/valid/test 全部 33984 张）、推理、`cost.py` 核验、打包提交。
> **不能**：重做 1024² 或其它分辨率的预处理（那需要 train 原图）。

### 档位 B · 推荐（8.9G）— 加上 valid 原图，能重做任意 split 的预处理

```bash
rsync -av --progress cxr_first/valid user@新设备:~/计挑赛/cxr_first/
```

### 档位 C · 完整（55G）— 连 train 原图一起搬

```bash
rsync -av --progress cxr_first/ user@新设备:~/计挑赛/cxr_first/
```

> 内网 100MB/s 约 9 分钟；外网 1MB/s 约 13 小时。**除非确定要换源分辨率，否则不要选 C。**

### 代码也可以走 git（不含数据）

```bash
# 本机
git -C /Users/mac/WorkBuddy/计挑赛 log --oneline   # 已有 2 次提交
# 新设备
git clone <你的远端> 计挑赛 && cd 计挑赛
# 然后把 baseline/data600、results/weights_7767、dl/*.pth、cxr_first/test 单独 rsync 过来
```

---

## 3. 新设备必须保持的目录结构

两套代码的默认路径都是**相对路径**，结构不变就**不需要配任何环境变量**：

```
计挑赛/
├── cxr_first/
│   ├── train.csv  valid.csv  test.csv
│   └── train/ valid/ test/  subject_*/study_*/pic_*.jpg
├── baseline/
│   └── data600/            ← 只有它，旧代码已删（见第 7 节）
│       ├── train/ valid/ test/
├── baseline1/              ← 主线代码（77.67）
├── baseline3/              ← 新框架
├── results/weights_7767/
├── dl/
└── *.md
```

对应关系（`baseline1/config.py`、`baseline3/config3.py` 第 26~31 行）：

| 常量 | 默认值 | 覆盖用的环境变量 |
|---|---|---|
| `DATA_ROOT` | `../cxr_first` | `CXR_DATA_ROOT` |
| `PREP_ROOT` | `../baseline/data600` | `CXR_PREP_ROOT` |
| `SRC_SIZE` | `600` | `CXR_SRC_SIZE` |

> **唯一会让路径失效的情况**：把 `baseline1/` 单独拷走、或把 `data600` 换位置。
> 真换了位置就显式传环境变量，别改代码默认值（默认值是全队共用的约定）。

---

## 4. 迁移后 5 条校验（2 分钟，缺一条都别开训）

```bash
cd ~/计挑赛

# ① 图片数量：两个都必须是 33984
find cxr_first -name "*.jpg" | wc -l
find baseline/data600 -name "*.jpg" | wc -l

# ② 代码语法（不需要 GPU）
python -m py_compile baseline1/*.py baseline3/*.py baseline3/vendor/*.py && echo OK

# ③ 架构可建 + 体积合规（不需要权重）
cd baseline3 && python tools/selftest_archs.py --img-size 392

# ④ 权重哈希（三个都必须对上）
shasum -a 256 results/weights_7767/full_base.pth \
              results/weights_7767/full_effv2m.pth \
              dl/convnext_base_22k_1k_224.pth
```

期望：

```
fefc03f0…5805  full_base.pth            (334M, convnext_base, 77.67 成员 A)
1ac00a2e…6873  full_effv2m.pth          (203M, efficientnet_v2_m, 77.67 成员 B)
8afbd1fa…0453  convnext_base_22k_1k_224.pth  (338M, 22k 起点)
```

```bash
# ⑤ 冒烟：6 张图跑一次推理（要有原图，所以必须搬了 cxr_first/test）
cd baseline1
python predict.py --split test --limit 6 --ckpt ../results/weights_7767/full_base.pth \
  --out /tmp/smoke.npz
```

看到 `probs.shape=(6, 10)` 就说明「代码 + 数据 + 权重 + 环境」四者都通了。

---

## 5. 复现：从训练到提交

### 5.1 baseline1（产出 77.67 的那套）

```bash
cd baseline1

# ① 自监督（规则补充四允许；30k 图 50 轮即饱和，别跑 200）
python -u ssl_pretrain.py --epochs 50 --img-size 384 --batch-size 128 \
  --out runs/ssl/simclr50.pth

# ② 两个架构各训一次（全量 train，官方 valid 监控）
python -u train.py --epochs 12 --img-size 448 --ssl runs/ssl/simclr50.pth --out runs/full_base
python -u train.py --backbone efficientnet_v2_m --epochs 12 --img-size 448 --out runs/full_effv2m

# ③ 出预测 → 融合 → 提交
python -u predict.py --split test --ckpt runs/full_base/best.pth   --out runs/pt_a.npz
python -u predict.py --split test --ckpt runs/full_effv2m/best.pth --out runs/pt_b.npz
python ensemble.py --split test --inputs runs/pt_a.npz runs/pt_b.npz --out submission_ens.csv

# ④ 合规核验（必须 0 张不一致 + <100ms）
python -u cost.py submission_ens.csv pictures_100 pictures_5 \
  --ckpt runs/full_base/best.pth runs/full_effv2m/best.pth
```

> `pictures_100/`、`pictures_5/` 是 100 张真原图副本，迁移时没带就重建（见 5.3）。

### 5.2 baseline3（要继续训练时用这套，已修掉 baseline1 的 8 个坑）

```bash
cd baseline3

# ① 探针（2 轮，20 分钟）—— 判断某个起点/配方值不值得全量
python -u train3.py --arch convnext_base_in22k --img-size 448 --batch-size 16 \
  --lr 1e-4 --probe --pretrained weights/convnext_base_22k_1k_224.pth \
  --train-csv ../baseline1/runs/folds/fold_0_train.csv \
  --val-csv   ../baseline1/runs/folds/fold_0_val.csv --out runs/probe_22k

# ② 全量（12 轮）
python -u train3.py --arch convnext_base_in22k --img-size 448 --batch-size 16 \
  --lr 1e-4 --epochs 12 --pretrained weights/convnext_base_22k_1k_224.pth \
  --out runs/full_22k

# ③ 推理 → 融合 → 提交 → 核验
python -u predict3.py --split test --ckpt runs/full_22k/best.pth --out runs/pt_22k.npz
python -u ensemble3.py --npz runs/pt_22k.npz runs/pt_b.npz --method mean --gate 1.0 \
  --out submission_v3.csv
python -u cost3.py submission_v3.csv ../baseline1/pictures_100 ../baseline1/pictures_5 \
  --ckpt runs/full_22k/best.pth --gate 1.0
```

> `--gate` 是 Normal Gating（类 10「无异常」与其余 9 类互斥，用它压低异常类概率），
> 尚未验证，默认 0 表示关闭；`--method` 只支持 `mean` / `logit`（`rank` 不合规，已移除）。

### 5.3 如果 pictures_100 没搬过来（3 分钟重建）

```bash
cd baseline1 && python - <<'EOF'
import os, shutil
from common import read_split_csv, raw_img_path
for i, r in enumerate(read_split_csv("test")[:105]):
    dst = "pictures_5" if i < 5 else "pictures_100"
    d = os.path.join(dst, r["Subject_id"], r["Study_id"]); os.makedirs(d, exist_ok=True)
    shutil.copy(raw_img_path(r, "test"), os.path.join(d, r["Dicom_id"] + ".jpg"))
print("ok")
EOF
```

---

## 6. 文档地图（读哪份、什么时候读）

| 文档 | 什么时候读 |
|---|---|
| **README_迁移与复现.md**（本文） | 迁移、部署、跑第一条命令 |
| `规则.md` / `规则补充.md` / `赛题说明.md` | **动手前必读**。预训练权重、体积、耗时、提交口径的唯一依据（口头答复不作数） |
| `项目状态与变更日志.md` | 想知道"现在到哪了、每个数字怎么来的、还有哪些坑" |
| `规则约束与应对复盘_1106.md` | 想知道"规则封了什么、我们怎么绕、哪些路已判负"（14 条实验总账 + 6 条教训） |
| `训练规划_baseline3_1105.md` | 用新框架训练时的完整规格（架构注册表、探针协议、权重下载、A1~A6 申报） |
| `baseline1/README_TRAIN.md` | baseline1 的逐条命令手册（612 行，最细） |
| `baseline1/README_COMPLIANCE.md` | 交付材料与合规申报（A1~A6 怎么写） |
| `archive/` | 历史文档，只在追溯旧决策时看 |

---

## 7. 这次删了什么、为什么（防止以后"我明明记得有"）

| 删除项 | 体积 | 理由 | 想恢复怎么办 |
|---|---|---|---|
| `cxr_first(3).zip` | 55G | 内容与 `cxr_first/` **完全同源**（zip 内 33984 张 jpg，逐项对上），纯重复备份 | 重新向主办方/队友要 |
| `data1024/` | 6.7G | 1024² 换源实验：valid +0.0024 但推理 +40ms，**净亏**，路线已关闭 | 用 `baseline1/preprocess.py --size 1024` 从原图重建（15 分钟） |
| `baseline/third_party/` | 2.4G | CheXFound 权重与代码，规则第 1 条明令禁止（医学预训练） | **不要恢复**（恢复即违规） |
| `baseline/` 其余代码 + 5 份 README | ~100K | 旧**非合规** pipeline（EVA-X + CheXFound），已被 baseline1 取代；`preprocess.py` 等功能 baseline1 都有且更新 | git 历史 `2b8575f` 可取回 |
| `baseline2/` | 36K | Kaggle TF/Keras notebook，框架不符、不可迁移 | git 历史 |
| `Grand-X-Ray-Slam-competition/` | 1.8M | Kaggle 亚军方案参考，配方已吸收进文档，权重不可用 | git 历史 |
| `results/cxr_deliver*` | 1.35G | T4 核验包与其压缩副本；内容都能用 `pack_t4.sh` 重新生成 | `bash pack_t4.sh` |
| `results/submission1103.csv` | 526K | 76.88 那次提交（与 76.91 实质相同，已无参考价值） | — |
| 历史文档 5 份 | 80K | 两周计划 / 执行手册 / 两次老师面谈材料 / Kaggle 检查清单，均已被新文档取代 | `archive/` |

**保留但需注意的两项**：
- `baseline/data600/` 现在**只有数据、没有代码**（旧代码已删），目录名保留是为了不动两套代码的相对路径默认值。
- `baseline1/distill.py` 保留：蒸馏本身没跑（5 折集成 +0.0021 太低），但**如果双模型过不了 100ms，它是把两个模型压成一个的唯一手段**，别删。

---

## 8. 迁移后要接着做的三件事（按优先级，判据都写死了）

| # | 事项 | 为什么 | 判据 | 成本 |
|---|---|---|---|---|
| 1 | **双模型 T4 实测** | 决定 77.67 是否可交付（附录 A 要求包内 CSV 是历史最高分，且 `cost.py` 必须能复现） | ≤100ms 定稿；100~150 有一次整改；>150 弃双模型 | 30 分钟 |
| 2 | **IN-22k 起点** | 唯一没验证的提分项，同架构同配方只换起点，单变量最干净 | 探针 ep2 相对 `convnext_base` 基准 **≥+0.008** 才全量 | 探针 20 分钟 + 全量 1 小时 |
| 3 | **Normal Gating + 融合方式** | 零成本、不改权重、不增耗时 | valid **≥+0.001** 即采纳 | 各 30 秒 |

**已经判负、不要再试**：训练分辨率（600 源内无效）、SSL 加长（50 轮饱和）、study 聚合换 max、
pos_weight 加码、同架构 5 折/多 seed 集成、6 路 TTA、DINOv2 起点（−0.035）、
类别平衡采样（−0.0086）、ASL（−0.0045）、batch8 不缩放 LR（−0.0067）、data1024（净亏）、
ConvNeXt-V2（CC BY-NC，违规）。

---

## 9. 一条容易被忘的纪律

**实验前先写判据，实验后按判据读。** 官方 valid 只有 859 个检查、噪声 ±0.01，
拿它做选型等于掷骰子——要用患者级 5 折（±0.0044）+ 配对 bootstrap 判差值。
我们已经在"看着好一点就上"这件事上浪费过三天。
