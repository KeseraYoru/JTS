# 训练流程手册：双模型 → 集成 → 提交 → 下载回本地

> 本文是**每次跑完整实验的操作清单**（EVA-X + CheXFound 两个模型训完、集成、出提交文件）。  
> 环境搭建、数据上传等一次性准备见 [`README_GPU.md`](README_GPU.md)，本文只管「开跑之后」。
>
> **总耗时约 1 小时**：EVA-X 10 epoch ≈ 27 分钟 → CheXFound 6 epoch ≈ 20 分钟 →  
> 推理+集成 ≈ 10 分钟 → 下载评估 5 分钟。

```
Step 0 准备（tmux）
   ↓
Step 1 训练 EVA-X（主模型，10 epoch，约 27 分钟）────┐
   ↓                                                │ 两个模型可并行（见 1.3）
Step 2 训练 CheXFound（第二模型，6 epoch，约 20 分钟）┘
   ↓
Step 3 两个模型各自推理（EVA-X 带 TTA，CheXFound 不带）→ 各自 npz
   ↓
Step 4 集成两个 npz → submission.csv（14140 行）
   ↓
Step 5 下载回 Mac（小文件打包）
   ↓
Step 6 本地评估 + 提交比赛平台
```

---

## Step 0. 准备（每次训练前）

```bash
ssh lcy@192.168.136.64
source ~/cxr/venv/bin/activate
cd ~/cxr/baseline
ls ~/cxr/cxr_first/*.csv | wc -l    # 期望 3；不是 3 见 README_GPU.md 2.2
```

用 tmux 跑训练，SSH 断了也不白跑：

```bash
tmux new -s train        # 断线重连后：tmux attach -t train
```

---

## Step 1. 训练 EVA-X（主模型）

### 1.1 命令

```bash
python3 train.py --model evax --epochs 10
```

**为什么是 10 epoch**：首训 6 epoch 的曲线每轮都在涨（0.7277 → 0.8131，第 6 轮还 +0.0034），  
没走平就是欠训；10 epoch 约 27 分钟，预期 study AUC 0.82 左右。

### 1.2 盯什么（每个 epoch 打印一行）

```
Epoch 3/10 | loss 0.1691 | 图像级 AUC 0.8063 | study 级 AUC 0.7958 | 163s
```

| 信号          | 健康                                  | 异常处理                  |
| ----------- | ----------------------------------- | --------------------- |
| study 级 AUC | 逐 epoch 上升，第 1 个 epoch 通常 0.72~0.78 | 不升/下降 → 停，把日志贴出来查     |
| loss        | 从 ~~0.15 稳步降到 0.10~~0.14            | 卡在 0.2+ 不动 → 学习率或数据问题 |
| 每 epoch 耗时  | 150~200 秒                           | 突然变慢 → 有人在抢 GPU 或磁盘   |

跑完在 `runs/evax_<日期时间>/` 得到（文件说明见 README_GPU.md 4.3）：

- `ema_f1.pth` ~ `ema_f10.pth`（推理用）、`best.pth`（valid 最优轮）、`metrics.json`

### 1.3 （可选）两个模型并行训练

4 张卡闲着也是闲着，想省时间就各分两张卡同时跑（主流程是串行，并行只是加速）：

```bash
# 窗口 A
CUDA_VISIBLE_DEVICES=0,1 python3 train.py --model evax --epochs 10
# 窗口 B（tmux 新开一个窗口：Ctrl+B 再按 C，或另起一个 tmux 会话）
CUDA_VISIBLE_DEVICES=2,3 python3 train.py --model chexfound --epochs 6
```

并行时每张卡分到的 batch 减半，注意 `--batch-size` 相应减半（如 32→16）防 OOM。

---

## Step 2. 训练 CheXFound（第二模型）

### 2.1 首次使用先准备第三方代码和权重

Google Drive 在国内没法直接下载，标准做法是**本地（能访问谷歌的设备）下载好 4 个文件，再传到服务器**。分三步：

```bash
# ① 克隆代码仓库（服务器不通 GitHub 就在 Mac 上 clone 再 rsync 过去）
bash setup_chexfound.sh

# ② 把 4 个权重文件传到服务器的 third_party/CheXFound_Model/
#    （在 Mac 上执行；权重已在 Mac 本地的情况。约 2.5G，断了重跑同一条续传）
rsync -av --partial --progress baseline/third_party/CheXFound_Model/ lcy@192.168.136.64:~/cxr/baseline/third_party/CheXFound_Model/

# ③ 服务器上装依赖 + 一键检查（全绿才算就绪）
pip install fvcore open_clip_torch omegaconf scikit-multilearn torchmetrics iopath pandas transformers scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 check_chexfound.py
```

`check_chexfound.py` 会逐项检查：代码目录、4 个权重文件（存在 + 大小合理 + 是有效
torch 权重而非下载失败的网页错误页）、7 个 Python 依赖、`chexfound` 各子模块能否
导入——任何一项不行都会给出具体补救命令，全绿才开训。

三个容易踩的点（都是「pip 包名 ≠ import 名」或版本冲突的坑）：

- **`open_clip_torch` 的导入名是 `open_clip`**，`skmultilearn` 的 pip 包名是
  **`scikit-multilearn`**（直接 pip install skmultilearn 会报找不到包）；
- **xformers 不用装**：官方环境 pin 的 0.0.18 和 torch 2.14/CUDA 13 不兼容，
  而代码会自动回退到 PyTorch 原生 attention（只有 warning，功能无损）；
- 官方 `conda-extras.yaml` 里的 mmcv/mmsegmentation/cuml 是完整预训练用的，
  我们只微调 GLoRIA 头，都不需要。

**只有 EVA-X 一个模型也能出提交**，这个模型是提分用的，不必阻塞主流程。

### 2.2 命令

```bash
python3 train.py --model chexfound --epochs 6
```

与 EVA-X 的差异（代码里已处理好，列出以便理解日志）：

- **backbone 全程冻结，只训 GLoRIA 分类头**（所以 loss 降得慢、AUC 上限低一些都正常）；
- 输入 512×512，默认 batch 32；
- 6 epoch（Kaggle 原版配置，冻结 backbone 收敛快，不需要像 EVA-X 那样加轮数）。

产出：`runs/chexfound_<日期时间>/ema_f1.pth` ~ `ema_f6.pth` + `best.pth` + `metrics.json`。

---

## Step 3. 两个模型各自推理

把下面的 `RUN_EVAX` / `RUN_CHEX` 换成实际的 runs 目录名（`ls runs/` 看）。

**取哪两个权重：用每个 run 的「最后两个 epoch」**（Kaggle 的做法是平均最后两轮 EMA 权重）。
先确认 epoch 数，再据此定文件名，别照抄示例：

```bash
# 看某个 run 训了多少个 epoch + 每轮的 study AUC
python3 -c "
import json
m = json.load(open('runs/evax_20261021_084833/metrics.json'))
print('完成 epoch 数:', len(m))
for e in m: print('  epoch', e['epoch'], 'study_auc', e['study_auc'])
"
ls runs/evax_20261021_084833/          # 看 ema_fN.pth 最大到几
```

| run 训了多少 epoch | 用哪两个 |
|---|---|
| 6 epoch（CheXFound、早期 EVA-X） | `ema_f5.pth ema_f6.pth` |
| 10 epoch（推荐的 EVA-X 配置） | `ema_f9.pth ema_f10.pth` |

```bash
RUN_EVAX=runs/evax_20261019_100000            # ← 换成你的
RUN_CHEX=runs/chexfound_20261019_103000       # ← 换成你的

# EVA-X：最后两个 epoch 权重 + TTA（Kaggle 原版配置；10 epoch 版见上表换 f9/f10）
python3 predict.py --model evax --split test --tta --ckpt $RUN_EVAX/ema_f5.pth $RUN_EVAX/ema_f6.pth --out $RUN_EVAX/pred_test.npz
python3 predict.py --model evax --split valid --tta --ckpt $RUN_EVAX/ema_f5.pth $RUN_EVAX/ema_f6.pth --out $RUN_EVAX/pred_valid.npz

# CheXFound：最后两个 epoch 权重，不用 TTA（Kaggle 原版配置）
python3 predict.py --model chexfound --split test --ckpt $RUN_CHEX/ema_f5.pth $RUN_CHEX/ema_f6.pth --out $RUN_CHEX/pred_test.npz
python3 predict.py --model chexfound --split valid --ckpt $RUN_CHEX/ema_f5.pth $RUN_CHEX/ema_f6.pth --out $RUN_CHEX/pred_valid.npz
```

预期：每个 npz 保存时打印 `probs.shape=(2218, 10)`（test）或 `(1422, 10)`（valid）；  
valid 推理会顺带打印该模型的 study 级 Macro AUC——**先记下这两个数**，集成后对比。

> 为什么要推 valid：下载回 Mac 后要算集成效果的分数，没有 valid npz 就没法本地评分。

---

## Step 4. 集成 + 生成提交文件

**先扫权重再出提交**（两个模型实力不均衡时，等权平均可能反而低于最强单模型）：

```bash
# 几秒出结果，不占 GPU；w_A=1.00 那行就是「只用 EVA-X」的分数，w_A=0.00 是「只用 CheXFound」
python3 scan_weights.py --split valid \
    --inputs $RUN_EVAX/pred_valid.npz $RUN_CHEX/pred_valid.npz
```

挑一个落在**平台区间**（不是孤立尖峰）的权重组合，然后：

```bash
python3 ensemble.py --split test --inputs $RUN_EVAX/pred_test.npz $RUN_CHEX/pred_test.npz \
    --weights 0.35 0.65 --out submission.csv      # 权重换成扫描的最优值
```

如果扫描显示最优是 `w_A=0.00`（即 EVA-X 没有加分），直接只用 CheXFound 出提交即可。

预期输出：

```
读取 runs/.../pred_test.npz (权重 0.500)
读取 runs/.../pred_test.npz (权重 0.500)
图像数 2218 -> study 数 1414（聚合方式 mean）
已写入 submission.csv（14140 行，全 10 类）
提交文件自检通过（study 覆盖、行数、类别 id、概率范围均正确）
```

- 14140 = 1414 个 study × 10 类；自检不过就别提交，按提示修；
- 只想先交单模型：`--inputs` 只写一个 npz 即可；
- 提交口径若主办方确认「只交 >0.5 的类」：改 `config.py` 的 `SUBMIT_ALL_CLASSES = False` 后重跑本步。

---

## Step 5. 下载回本地（Mac）

大权重（几个 GB）留在服务器，只搬回小文件。先在**服务器**上打包：

```bash
tar czf ~/results.tgz $RUN_EVAX/metrics.json $RUN_EVAX/pred_valid.npz $RUN_CHEX/metrics.json $RUN_CHEX/pred_valid.npz submission.csv
```

在 **Mac** 上下载并解包：

```bash
cd /Users/mac/WorkBuddy/计挑赛/baseline
scp lcy@192.168.136.64:~/results.tgz .
mkdir -p runs && tar xzf results.tgz
```

---

## Step 6. 本地评估 + 提交

### 6.1 本地评分（Mac，无需 GPU、无需图像，秒出）

```bash
# 单模型分数（复现推理时的打印）
python3 ensemble.py --split valid --inputs runs/evax_xxx/pred_valid.npz --out /tmp/v1.csv
python3 ensemble.py --split valid --inputs runs/chexfound_xxx/pred_valid.npz --out /tmp/v2.csv

# 集成分数（关键数字：应该 ≥ 两个单模型）
python3 ensemble.py --split valid --inputs runs/evax_xxx/pred_valid.npz runs/chexfound_xxx/pred_valid.npz --out /tmp/v_ens.csv
```

读法：Macro AUC 是主指标，十类等权——优先补最弱的类；两个模型差 0.005 以内  
不要下结论（valid 只有 859 个 study，罕见类正样本个位数）。

### 6.2 提交平台

1. 上传 `submission.csv`（UTF-8；**别用 Excel 打开再保存**，编码和格式会坏）；
2. **每日限 2 次**，团队指定一人管理名额；
3. 节奏：第一次交 baseline 占位 → 之后只在 valid 有实质提升（+0.005 以上）时才交。

### 6.3 落实验台账（团队规则）

| 日期    | runs 目录              | 模型/改动        | valid study AUC | 最弱 3 类 | 提交回分 |
| ----- | -------------------- | ------------ | --------------- | ------ | ---- |
| 10-19 | evax_20261019_093551 | EVA-X 6ep 基线 | 0.8131          | 1/8/4  | —    |

---

## 附 A. 完整命令清单（按顺序，替换 runs 目录名后可直接粘贴）

```bash
# ---------- 服务器 ----------
source ~/cxr/venv/bin/activate && cd ~/cxr/baseline
tmux new -s train

# Step 1：EVA-X 10 epoch（约 27 分钟）
python3 train.py --model evax --epochs 10

# Step 2：CheXFound 6 epoch（约 20 分钟；首次先按 2.1 准备：clone 代码 + 传权重 + check_chexfound.py 全绿）
python3 train.py --model chexfound --epochs 6

# Step 3：推理（RUN_* 换成 ls runs/ 看到的实际目录）
# ⚠ ckpt 取每个 run 的「最后两个 epoch」：6 epoch 用 f5+f6，10 epoch 用 f9+f10（见 Step 3 表格）
RUN_EVAX=runs/evax_xxxxxx; RUN_CHEX=runs/chexfound_xxxxxx
python3 predict.py --model evax --split test --tta --ckpt $RUN_EVAX/ema_f5.pth $RUN_EVAX/ema_f6.pth --out $RUN_EVAX/pred_test.npz
python3 predict.py --model evax --split valid --tta --ckpt $RUN_EVAX/ema_f5.pth $RUN_EVAX/ema_f6.pth --out $RUN_EVAX/pred_valid.npz
python3 predict.py --model chexfound --split test --ckpt $RUN_CHEX/ema_f5.pth $RUN_CHEX/ema_f6.pth --out $RUN_CHEX/pred_test.npz
python3 predict.py --model chexfound --split valid --ckpt $RUN_CHEX/ema_f5.pth $RUN_CHEX/ema_f6.pth --out $RUN_CHEX/pred_valid.npz

# Step 4：集成出提交
python3 ensemble.py --split test --inputs $RUN_EVAX/pred_test.npz $RUN_CHEX/pred_test.npz --out submission.csv

# Step 5：打包小结果
tar czf ~/results.tgz $RUN_EVAX/metrics.json $RUN_EVAX/pred_valid.npz $RUN_CHEX/metrics.json $RUN_CHEX/pred_valid.npz submission.csv

# ---------- Mac ----------
cd /Users/mac/WorkBuddy/计挑赛/baseline
scp lcy@192.168.136.64:~/results.tgz . && tar xzf results.tgz
scp lcy@192.168.136.64:~/cxr/baseline/submission.csv .     # 提交文件也拿一份

# Step 6：本地评估集成效果
python3 ensemble.py --split valid --inputs runs/evax_xxxxxx/pred_valid.npz runs/chexfound_xxxxxx/pred_valid.npz --out /tmp/v_ens.csv
```

## 附 B. 常见问题

| 症状                                          | 解法                                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| `FileNotFoundError: ../cxr_first/train.csv` | 3 个 CSV 没传，见 README_GPU.md 2.2 第 ② 条                                     |
| CheXFound 第 2 步崩溃：`Expected to have finished reduction ... did not receive grad` | chexfound 库会初始化 NCCL 进程组并把 glori 头包进 DDP，与我们的 40→10 类头改造冲突；2026-10-22 已在 `models/chexfound.py` 修复（从源头禁止库初始化进程组 + 拆 DDP 双保险）。**先确认服务器代码是最新的**：`grep -c _suppress_library_distributed models/chexfound.py` 应输出 2；不是 2 就先重传代码再跑 |
| CheXFound 起不来 | 跑 `python3 check_chexfound.py`，按它给的补救命令逐项修（代码目录/4 个权重文件/9 个 pip 包） |
| 训练 OOM                                      | `--batch-size 16`；并行训练时记得减半                                              |
| 推理比训练慢很多                                    | 正常，TTA 要推 6 遍；CheXFound 无 TTA 会快很多                                       |
| 集成后 AUC 反而比单模型低                             | 两个 npz 顺序/模型不匹配，或某个模型有问题；先各自单评再合                                         |
| 提交文件被平台拒绝                                   | 用 Excel 打开过 → 编码坏了；重新用本文命令生成                                             |
