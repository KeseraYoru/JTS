# 迁移与复现 · 入口文档

> 最后更新：2026-11-08　｜　总占用 **57G**，其中 52G 是原始影像，**可丢弃的冗余约 1.85G**
> 这份文档是迁移时的唯一入口：先读第 1、2 节搬东西，再按第 4 节校验，第 5 节跑训练。
>
> **一条命令搞定迁移**：`bash migrate.sh user@新设备:~/计挑赛`
> （默认含全部原图 52G；加 `SKIP_RAW=1` 跳过 train+valid 的 48G，只搬约 6.9G，test 原图仍保留）

---

## 0. 一句话现状

平台最高 **77.67**（full_base + effv2m 双架构集成，277MB，5090 上 39ms → T4 估 ≈100ms，**未实测**）。
11-08 又搜出更好的候选：**in22k30 + swinb30**（valid 0.8112，334MB，估 T4 92ms，
唯一一对 bootstrap 置信区间全为正的），以及 in22k30 + effv2s30（0.8102，208MB）。
复赛线 77.7，一等奖线 78.5。

**当前最大风险不是分数，是「体积 × 耗时」能否同时过关**：
官方 T4 上双模型跑不跑得进 100ms，一次实测就能定生死（`cost.py`）。
其余 12 条提分路已用实验判负，清单见第 8 节。

---

## 1. 要搬什么、不搬什么

### 1.1 必须搬（合计约 55G，其中 52G 是原图）

| 路径 | 体积 | 是什么 | 不搬的后果 |
|---|---|---|---|
| `cxr_first/train/` | 46G | 30344 张训练原图 | 不能重做 1024²/384 等换分辨率预处理（训练本身不受影响） |
| `cxr_first/valid/` | 2.1G | 1422 张验证原图 | 不能重做 valid 预处理 |
| `cxr_first/test/` | 3.4G | 2218 张测试原图 | **`cost.py` 核验跑不了**（它要真原图计时） |
| `cxr_first/*.csv` | 1.9M | 三个标注 CSV（比赛的"标签真值"） | 训练直接起不来 |
| `baseline/data600/` | 2.5G | 33984 张 600² 预处理图 | 得花 15 分钟重跑 `preprocess.py` |
| `baseline1/` | 588K | 产出 77.67 的主线代码（含规则、许可证、17 个 py） | 主线没了 |
| `baseline3/` | 420K | 新框架：换起点（IN-22k / DINOv2）+ 修掉 baseline1 的 8 个坑 | 继续训练用不了 |
| `results/weights_7767/` | 537M | 77.67 的两个模型权重（fp32） | **保底提交物没了**，得重训 |
| `results/runs/` | 5.9M | 各实验的 metrics / npz 证据（附 A 申报要用） | 申报材料缺证据 |
| `results/submission_*.csv` | 2.6M | 5 份历史提交（77.67 / 76.91 / 22k_swin 等） | 说不清提交历史 |
| `dl/convnext_base_22k_1k_224.pth` | 338M | ImageNet-22k 起点权重 | 要重下 354MB（且要能连上） |
| `baseline1/pictures_100` + `pictures_5` | 149M | `cost.py` 计时用的 105 张真原图 | 得按 5.3 重建（要 test 原图） |
| `tools/screen_solution.py` | 28K | 合规红线扫描脚本 | 少一道自检 |
| `*.md`（含 `规则.md`/`规则补充.md`/`赛题说明.md`） | ~120K | 比赛规则与全部决策记录 | **等于把判罚依据扔了** |
| `archive/` | 80K | 历史文档（两周计划、执行手册、老师简报） | 只是追溯用得上，可选 |

### 1.2 不用搬（合计约 1.85G，全都能重建）

| 路径 | 体积 | 为什么不用搬 | 想恢复怎么办 |
|---|---|---|---|
| `results/cxr_t4.tgz` | 647M | T4 核验包的压缩副本 | `bash pack_t4.sh` |
| `results/cxr_t4.zip` | 647M | 上面那份的第二份压缩，纯重复 | 同上，别做两份 |
| `results/cxr_t4/weights/` | 552M | 与 `results/weights_7767/` **逐字节相同**（sha256 已核） | 直接拷过去 |
| `results/cxr_t4/baseline1/` | 4.2M | `baseline1/` 的完整副本 | 就是 `baseline1/` 本身 |
| `__pycache__/`（42 个 .pyc） | ~400K | 首次运行自动重建 | 无需 |
| `.DS_Store` | 12K | macOS 目录元数据 | 无需 |
| `.git/` | 788K | 新设备 `git clone` 即可（已从 322M 瘦到 788K） | 第 10 节 |

> **注意**：`results/cxr_t4/` 里的 `pictures_100`/`pictures_5` 是**唯一存活副本**
> （`baseline1/pictures_*` 已不存在），所以它们属于"必须搬"，别整个 t4 目录一起删。

---

## 2. 迁移：一条命令，或手动 rsync

### 2.1 推荐：跑脚本（排除清单写死在脚本里，不靠记忆）

```bash
cd /Users/mac/WorkBuddy/计挑赛

# ① 先演练，只看会传什么，一个字节都不传
DRY=1 bash migrate.sh user@新设备:~/计挑赛

# ② 正式传（含 52G 原图）
bash migrate.sh user@新设备:~/计挑赛

# ③ 带宽紧就跳过 train+valid 共 48G 原图（只传约 6.9G，训练+核验照样能跑）
SKIP_RAW=1 bash migrate.sh user@新设备:~/计挑赛

# 目标端是移动硬盘/U 盘时，直接写路径
bash migrate.sh /Volumes/移动硬盘/计挑赛
```

脚本按 **代码 → data600 → 权重 → 计时用图 → 原图** 五段传，
先传完代码那几秒就能在新设备做语法/架构校验；中断了重跑即可续传（`--partial`）。
传完会打印新设备上还需手动做的 3 条收尾命令（建软链、重建五折、校验）。

### 2.2 手动 rsync（脚本跑不了时的备用）

```bash
cd /Users/mac/WorkBuddy/计挑赛

# 代码 + 文档 + 标注 CSV + 小证据（几 MB）
# 注意：只排除 cxr_first 的三个 split 子目录，别排掉整个 cxr_first/（那会连 train.csv 一起排掉，训练起不来）
rsync -av --progress \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='cxr_first/train/' --exclude='cxr_first/valid/' --exclude='cxr_first/test/' \
  --exclude='baseline/data600/' --exclude='dl/' \
  --exclude='results/weights_7767/' --exclude='results/cxr_t4/' \
  ./ user@新设备:~/计挑赛/

# 预处理数据 + 权重（3.4G）
rsync -av --progress baseline/data600/            user@新设备:~/计挑赛/baseline/data600/
rsync -av --progress results/weights_7767/         user@新设备:~/计挑赛/results/weights_7767/
rsync -av --progress dl/                           user@新设备:~/计挑赛/dl/

# 计时用图（149M，注意源在 cxr_t4 里，目标是 baseline1 下）
rsync -av --progress results/cxr_t4/pictures_100/ user@新设备:~/计挑赛/baseline1/pictures_100/
rsync -av --progress results/cxr_t4/pictures_5/   user@新设备:~/计挑赛/baseline1/pictures_5/

# 原始数据（52G，最慢，放最后；只要 test 就改成 cxr_first/test/）
rsync -av --progress cxr_first/                    user@新设备:~/计挑赛/cxr_first/
```

> 内网 100MB/s 传 52G 约 9 分钟；外网 1MB/s 约 13 小时。

### 2.3 代码走 git（数据仍走 rsync）

```bash
# 本机：先推到 GitHub（仓库已瘦身到 788K，见第 10 节）
bash push.sh

# 新设备
git clone https://github.com/KeseraYoru/JTS.git 计挑赛 && cd 计挑赛
# 再用上面的 rsync 把 data600 / weights_7767 / dl / cxr_first / pictures 补过来
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
│   ├── runs/folds/         ← 五折划分 CSV（已丢失，须重建，见第 4 节 ⑥）
│   ├── pictures_100/       ← cost.py 计时用 100 张
│   └── pictures_5/         ← 5 张预热
├── baseline3/              ← 新框架
├── results/
│   ├── weights_7767/       ← full_base.pth + full_effv2m.pth
│   ├── runs/               ← 各实验 metrics / npz
│   └── submission_7767_ens.csv
├── dl/                     ← convnext_base_22k_1k_224.pth
├── migrate.sh  push.sh
└── *.md
```

对应关系（`baseline1/config.py` 第 17~21 行、`baseline3/config3.py` 第 26~31 行）：

| 常量 | 默认值 | 覆盖用的环境变量 |
|---|---|---|
| `DATA_ROOT` | `../cxr_first` | `CXR_DATA_ROOT` |
| `PREP_ROOT` | `../baseline/data600` | `CXR_PREP_ROOT` |
| `SRC_SIZE` | `600` | `CXR_SRC_SIZE` |

> **唯一会让路径失效的情况**：把 `baseline1/` 单独拷走、或把 `data600` 换位置。
> 真换了位置就显式传环境变量，别改代码默认值（默认值是全队共用的约定）。

---

## 4. 迁移后 7 条校验（3 分钟，缺一条都别开训）

```bash
cd ~/计挑赛

# ① 图片数量：三个都必须对上
find cxr_first -name "*.jpg" | wc -l          # 33984（搬了原图才查这条）
find baseline/data600 -name "*.jpg" | wc -l   # 33984
find baseline1/pictures_100 baseline1/pictures_5 -name "*.jpg" | wc -l   # 105

# ② 代码语法（不需要 GPU）
python -m py_compile baseline1/*.py baseline3/*.py baseline3/vendor/*.py && echo OK

# ③ 架构可建 + 体积合规（不需要权重）
cd baseline3 && python tools/selftest_archs.py --img-size 392 && cd ..

# ④ 权重哈希（三个都必须对上）
shasum -a 256 results/weights_7767/full_base.pth \
              results/weights_7767/full_effv2m.pth \
              dl/convnext_base_22k_1k_224.pth

# ⑤ 重建五折划分（原 CSV 已丢失，种子 48 可复现同一份）
cd baseline1 && python make_folds.py --k 5 --seed 48 --out runs/folds && cd ..
ls baseline1/runs/folds/ | head   # 应有 fold_0_train.csv / fold_0_val.csv … fold_4_*

# ⑥ 恢复 pack_t4.sh 的权重路径 + 提交文件（软链，不额外占空间）
cd baseline1
mkdir -p runs/full_base runs/full_effv2m
ln -sf ../../../results/weights_7767/full_base.pth   runs/full_base/best.pth
ln -sf ../../../results/weights_7767/full_effv2m.pth runs/full_effv2m/best.pth
cp ../results/submission_7767_ens.csv submission_ens.csv
cd ..
```

期望：

```
fefc03f0…5805  full_base.pth            (334M, convnext_base, 77.67 成员 A)
1ac00a2e…6873  full_effv2m.pth          (203M, efficientnet_v2_m, 77.67 成员 B)
8afbd1fa…0453  convnext_base_22k_1k_224.pth  (338M, 22k 起点)
```

```bash
# ⑦ 冒烟：6 张图跑一次推理（要有原图，所以必须搬了 cxr_first/test）
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
| `migrate.sh` | **换设备就运行它**。五段式传输，排除清单写死，中断可续传 |
| `push.sh` | 推代码到 GitHub 前运行它。会拦住任何 >50M 的对象 |
| `.gitignore` | 想知道"什么该进 git"，看它一页就够 |
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

---

## 10. git 与 push（2026-11-08 修过一次大坑）

### 10.1 曾经的问题

`dl/convnext_base_22k_1k_224.pth`（338M）被提交进了版本库，导致 `.git` 膨胀到 **322M**。
GitHub 单文件硬上限 100M，这个仓库**永远 push 不上去**，会报：

```
remote: error: File dl/convnext_base_22k_1k_224.pth is 338.00 MB; this exceeds GitHub's file size limit of 100.00 MB
```

已做的处理（当天完成，`.git` 从 322M → **788K**）：

```bash
git rm -r --cached dl baseline3/__pycache__ ...   # 1. 从索引移除
# 2. 重写 .gitignore
git commit -m "chore: 权重与缓存不入库"
git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch dl' \
  --prune-empty --tag-name-filter cat -- --all      # 3. 从整条历史里剔除该 blob
git reflog expire --expire=now --all && git gc --prune=now
```

> 改动前的 `.git` 完整备份在 `/tmp/git_backup_JTS_20261108_2205`（316M）。
> 三条提交历史都保留了，只有那个 338M 的 blob 被拿掉。

### 10.2 现在的规矩

| 入库 | 不入库（走 rsync，见 `.gitignore`） |
|---|---|
| 全部 `.py` / `.sh` / `.md` | `cxr_first/` 原图（52G） |
| `baseline1/licenses/`（附录 A 要许可证全文） | `baseline/data600/` |
| `results/submission_7767_ens.csv`（保底提交物，唯一放行的 results 文件） | `*.pth` / `*.pt` / `*.npz` |
| `.workbuddy/memory/`（决策记录） | `results/*` 其它、`runs/`、`pictures_*` |
| | `*.zip` / `*.tgz`、`__pycache__/`、`.DS_Store` |

### 10.3 推送

```bash
cd /Users/mac/WorkBuddy/计挑赛
bash push.sh                 # 体检 → 提交 → 预检连通 → 推 origin master
CHECK_ONLY=1 bash push.sh    # 只体检：待提交和整个历史里有没有 >50M 的对象
```

脚本会在推送前拦住任何超限对象——**这个坑踩过一次就够了**。

**连不上 GitHub 时**（本机当前就是这个状态，`CONNECT tunnel failed 502`）：
先开代理，再给 git 配上（Clash 默认 7890，V2Ray 常见 1080）：

```bash
git config --global http.proxy  http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
# 用完关掉
git config --global --unset http.proxy && git config --global --unset https.proxy
```

或者改用 SSH（前提：公钥已传到 GitHub）：

```bash
git remote set-url origin git@github.com:KeseraYoru/JTS.git
```

> **历史重写过了**：如果哪台设备之前 clone 过旧版本，直接 `git pull` 会冲突，
> 在新设备上重新 `git clone` 最省事。
