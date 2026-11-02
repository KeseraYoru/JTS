# GPU 训练机上机指南（上传 → 训练 → 下载回评估 全流程）

> **适用场景**：数据和预处理都在 Mac 上做好了，现在要把东西搬到学校机房的 GPU 机上
> 训练，再把结果搬回 Mac 评估、提交比赛。
> Mac 端的准备工作见 [`README_MAC.md`](README_MAC.md)，方案原理见
> [`README.md`](README.md)，本文只讲「上机之后」的完整操作。

---

## 0. 全流程一览

```
Mac（已就绪）                     GPU 训练机                     Mac（评估）
─────────────                    ────────────                   ──────────
data600/ 2.5G ✓  ── rsync 上传 ──►  baseline/                     │
weights/  1.1G ✓  ── rsync 上传 ──►  data600/ weights/            │
代码 12 个 .py   ── rsync 上传 ──►  cxr_first/*.csv（仅 CSV）     │
CSV×3（2.5MB）   ── rsync 上传 ──►  装依赖 → 训练 20~40min       │
                                       │                          │
                                       ├─ runs/evax_*/            │
                                       │   ema_f5/ema_f6.pth      │
                                       │   metrics.json           │
                                       └─ pred_test.npz ──────────┼─ rsync 下载 ─►  ensemble 本地评分
                                                                  │                  ↓
                                                                  └─ submission.csv ◄┘  提交平台
```

预计总耗时：**上传 10 分钟 + 装环境 10~20 分钟 + 训练 20~40 分钟 + 推理 10 分钟 +
下载评估 5 分钟 ≈ 1 小时**（EVA-X 单模型，4× RTX 5090 DataParallel）。

---

## 1. 上传前检查（Mac 上，2 分钟）

```bash
cd /Users/mac/WorkBuddy/计挑赛/baseline

# ① 预处理数据完整性（期望 33984）
find data600 -name "*.jpg" | wc -l

# ② 预训练权重在位（期望约 1.1G）
ls -lh weights/eva_x_base_patch16_merged520k_mim.pt

# ③ 代码文件齐全（期望 12：9 个在根 + 3 个在 models/）
# 用 find 而不是 ls *.py：find 自己做匹配，不依赖 shell 通配符展开，
# 从文档复制命令时即使 * 被转义成 \* 也能正常工作
find . -maxdepth 2 -name "*.py" | wc -l
```

三项都对得上再上传。任何一项不对，先回 Mac 修好（数据不全就重跑
`preprocess.py`，已存在的文件会自动跳过，接着跑）。

> 小知识：shell 里的 `*` 是通配符，会被展开成真实文件名列表；写成 `\*` 则是
> 「一个字面量星号」，命令就会去找一个名叫 `*.py` 的文件，必然报
> `No such file or directory`。用 `find ... -name "*.py"` 可以绕开这个坑。

---

## 2. 上传到 GPU 机

### 2.1 假设与目录约定

- GPU 机和你同校园网，知道它的 IP（例如 `10.20.30.40`），你有 SSH 账号；
- 以下命令统一用 `~/cxr/` 作为 GPU 机上的工作目录，可按需改；
- **先建目录**（rsync 只会创建目标的最后一级，父目录不存在会直接报
  `mkdir failed` 退出）：`ssh 用户名@10.20.30.40 "mkdir -p ~/cxr"`。

### 2.2 上传（四条命令，都是单行，直接整行复制粘贴）

```bash
cd /Users/mac/WorkBuddy/计挑赛

# ① 代码和文档（秒级，以后改代码只重跑这一条）
# 注意：不要加 --partial！macOS 自带 rsync 2.6.9 上 --partial 会让
# --exclude 'dir/' 失效（实测会连 data600/weights 一起传），已移除
rsync -av --exclude 'data600/' --exclude 'weights/' --exclude 'runs/' baseline/ 用户名@10.20.30.40:~/cxr/baseline/

# ② 三个标注 CSV（约 2.5MB，秒传；服务器上没它们会报 FileNotFoundError: ../cxr_first/train.csv）
# 图片不用传，但 CSV 全服务器独一份，必须传
ssh 用户名@10.20.30.40 "mkdir -p ~/cxr/cxr_first"
rsync -av --partial cxr_first/train.csv cxr_first/valid.csv cxr_first/test.csv 用户名@10.20.30.40:~/cxr/cxr_first/

# ③ 预处理数据 2.5G（校园网几分钟；无 exclude，--partial 安全，断了重跑续传）
rsync -av --partial --progress baseline/data600/ 用户名@10.20.30.40:~/cxr/baseline/data600/

# ④ 预训练权重 1.1G（无 exclude，--partial 安全；避免在 GPU 机上冒 HF 下载失败的风险）
rsync -av --partial --progress baseline/weights/ 用户名@10.20.30.40:~/cxr/baseline/weights/
```

> 断线重传：① ② 都是小文件，断了原样重跑即可；③④ 带 `--partial`，断了原样
> 重跑会续传。如果机房机器不联网/不通 SSH，就用移动硬盘拷贝，目录结构照原样摆。
>
> **粘贴注意事项（实测踩坑）**：命令必须保持**单行**完整粘贴，`--exclude` 后面
> 的引号要保留。如果报 `link_stat ".../计挑赛/data600" failed`（路径里没有
> `baseline/`），说明 exclude 规则没生效——先检查是不是手滑加了 `--partial`，
> 或粘贴时参数错位，把命令改成一行重新粘贴。

### 2.3 上传后校验（GPU 机上执行）

```bash
cd ~/cxr/baseline
ls ~/cxr/cxr_first/*.csv | wc -l             # 期望 3（缺了会报找不到 train.csv）
find data600 -name "*.jpg" | wc -l           # 期望 33984
ls -lh weights/                              # 期望 1.1G 的 .pt
```

### 2.4 不要传的东西

| 不传 | 原因 |
|---|---|
| `cxr_first/` 里的 46G **图片** | 训练只读 600² 小图，传图片纯浪费带宽（同目录的 3 个 CSV 必须传，见 2.2 第 ② 条） |
| `cxr_first(3).zip`（51G） | 同上 |
| `baseline/runs/` | GPU 机上会重新生成；Mac 上的冒烟产物已删 |

---

## 3. GPU 机上环境安装（首次，约 15 分钟）

> Mac 上不存在、服务器上必须处理的四件事：**系统 Python 别污染、CUDA 版本要匹配、
> 外网可能不通、磁盘要够**。按下面顺序来。

### 3.1 先诊断：这台服务器什么情况

```bash
python3 --version                                        # ① Python 版本（要 3.10~3.12）
nvidia-smi                                               # ② 右上角 CUDA Version（见 3.4）
curl -sI -m 10 https://pypi.org/simple/ | head -1         # ③ 有外网吗（HTTP/2 200 = 通）
df -h ~ | tail -1                                        # ④ 家目录剩余空间（要 > 15G）
```

把 ② 的 `CUDA Version: XX.X` 记下来，3.4 节要用。③ 不通的话直接跳 3.5。

### 3.2 建虚拟环境（不要直接装进系统 Python）

服务器多半多人共用，系统 Python 里可能已有别人的包，版本一冲突就很难查。
自己的环境自己管：

```bash
python3 -m venv ~/cxr/venv
source ~/cxr/venv/bin/activate
# 激活后提示符前会出现 (venv)，之后每条命令都在这个环境里跑
```

- 报 `ensurepip is not available` → 系统缺组件：`sudo apt install python3-venv`
  （CentOS/RHEL 是 `sudo yum install python3-virtualenv`），或者用 conda：
  `conda create -n cxr python=3.12 -y && conda activate cxr`；
- 激活后 `which python3` 应指向 `~/cxr/venv/bin/python3`，不是 `/usr/bin/python3`。

### 3.3 装依赖（有外网的情况）

```bash
cd ~/cxr/baseline
pip install -U pip                       # 先升级 pip，老 pip 解析新包容易出错
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 清华源慢/挂时换中科大或阿里源：
# -i https://mirrors.ustc.edu.cn/pypi/simple   或   -i https://mirrors.aliyun.com/pypi/simple/
```

约 2~4 GB（torch 占大头），几分钟到十几分钟。**装完必须做 3.4 的检查**——
Linux 上默认装到的是带 CUDA 的 torch，但驱动太老会用不了。

### 3.4 torch 与 CUDA 的匹配（最容易翻车的一步）

看 3.1 里 `nvidia-smi` 右上角的 `CUDA Version`（这是**驱动支持的最高版本**）：

| 驱动 CUDA Version | 装法 | 说明 |
|---|---|---|
| **12.x / 13.x** | 默认 `pip install torch torchvision` 即可 | PyPI 的 Linux 版 torch 自带 CUDA 12/13 运行库，向下兼容。本服务器实测：驱动 580.126.09（CUDA 13.0）+ pip 默认 torch → `torch.cuda.is_available()=True` |
| **11.x** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` | 换成 CUDA 11.8 专用构建 |
| 更低/看不懂 | 把 `nvidia-smi` 完整输出贴给懂的人，别瞎试 | 驱动太老可能得换机器 |

验证 torch 真的能用 GPU（不是装成了 CPU 版）：

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# 必须看到 True 和一个 CUDA 版本号（如 12.x）；
# 若是 False 或 None，说明装成了 CPU 版 / CUDA 不匹配，见第 10 节
```

> 题面写的「PyTorch 2.10 + CUDA 12.8」是**复赛验证容器**的要求，不是训练机的。
> 训练机只要 torch 的 CUDA 构建 ≤ 驱动支持版本即可，不必追求和题面一致。

### 3.5 服务器没外网的备选（Mac 代下载）

校园机房常禁外网。在**能上网的 Mac** 上按服务器的平台下载 wheel，再传过去：

```bash
# Mac 上（注意 --python-version 要和服务器的 python3 --version 一致）
pip download -r requirements.txt \
    --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all: \
    -d ~/wheels -i https://pypi.tuna.tsinghua.edu.cn/simple
# torch 的 CUDA 版要约 2.5G，慢但能下；torch/torchvision 要版本配套

rsync -av --partial --progress ~/wheels/ 用户名@GPU机IP:~/wheels/     # 传过去

# 服务器上离线安装
source ~/cxr/venv/bin/activate
pip install --no-index --find-links ~/wheels -r ~/cxr/baseline/requirements.txt
```

这条路的坑：wheel 的平台/Python 版本/ CUDA 构建必须和服务器对得上。
**先试 3.3，实在没网再走这条。**

### 3.6 装完验证（三项全过再训练）

```bash
cd ~/cxr/baseline
nvidia-smi                                  # ① 能看到 GPU 和显存（本服务器：4x RTX 5090 32GB）
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
                                            # ② 应打印 True 和显卡名
python3 -c "
from common import read_split_csv
from dataset import ChestXrayDataset, make_val_tfms
rows = read_split_csv('valid')[:4]
ds = ChestXrayDataset(rows, 'valid', transform=make_val_tfms(448))
img, dicom_id, labels = ds[0]
print(img.shape, dicom_id, labels.shape)
"                                         # ③ 应打印 torch.Size([3,448,448]) pic_xxx torch.Size([10])
```

③ 同时验证了数据路径、图像解码、增强 pipeline、三件套返回结构，是最有价值的一项。

### 3.7 日常使用

```bash
source ~/cxr/venv/bin/activate     # 每次 SSH 登服务器第一件事（激活后才有 torch）
deactivate                          # 用完退出
```

环境搞坏了想重来：`rm -rf ~/cxr/venv` 再走一遍 3.2~3.3，五分钟恢复，别试图修。
`data600/`、`weights/`、代码都在 venv 外面，删了重来不丢任何东西。

第 ③ 项是关键的连通性测试：同时验证了数据路径、图像解码、增强、三件套返回结构。

---

## 4. 训练（核心步骤）

### 4.1 先花 1 分钟冒烟（强烈建议）

```bash
python3 train.py --model evax --limit 64 --epochs 1 --batch-size 4 --out runs/preflight
```

能跑完（出现 `Epoch 1/1` 和权重保存）说明全链路没问题，放心正式跑。
`runs/preflight/` 之后删掉。

### 4.2 正式训练 EVA-X

```bash
# 建议用 tmux 防 SSH 断线（训练跑几十分钟，断线就白跑了）
tmux new -s train     # 或 nohup python3 train.py --model evax > train.log 2>&1 &

python3 train.py --model evax
```

多卡说明：检测到多卡会自动用 DataParallel 把 batch 分到各卡（本服务器 4 卡）。
想强制单卡跑（排查问题用）：`CUDA_VISIBLE_DEVICES=0 python3 train.py --model evax`。

**每个 epoch 的正常输出长这样：**

```
设备: cuda | AMP: True | batch: 32 | epochs: 6 | seed: 48
使用 4 块 GPU（DataParallel）                ← 多卡自动生效
各类别 pos_weight: [9.66 7.54 ...]              ← 罕见类权重高，正常
[Train]: 100%|████| 948/948 [03:12<00:00]      ← 948 步 = 30344/32
[Val]: 100%|████| 18/18 [00:31<00:00]
Epoch 1/6 | loss 0.0921 | 图像级 AUC 0.8312 | study 级 AUC 0.8236 | 224s
  每类 AUC:  1:0.6123  2:0.7455 ... 10:0.9310
```

**要盯的三个数：**
1. `study 级 AUC` —— 选型依据，应该逐 epoch 上升（第 1 个 epoch 就可能到 0.80+）；
2. `loss` —— 稳步下降，如果在 0.5 以上卡住，学习率或数据有问题；
3. 每类 AUC —— 记住你最弱的 3 个类，后期优化对象。

**时间预期（4× RTX 5090）**：DataParallel 每 epoch 约 3~6 分钟，6 个 epoch 约
**20~40 分钟**（单卡约 12~20 分钟/epoch）。题面写的 T4 是复赛推理机的规格，
训练机快得多。显存不够（OOM）就 `--batch-size 16`；还想更快可 `--batch-size 64`。

### 4.3 训练产出文件（`runs/evax_日期_时间/`）

| 文件 | 大小 | 用途 |
|---|---|---|
| `ema_f1.pth` ~ `ema_f6.pth` | 各 345M | **推理用这个**（每 epoch 的 EMA 权重） |
| `best.pth` | 345M | valid study AUC 最好的那个 epoch 的 EMA 权重 |
| `model_f1.pth` ~ `model_f6.pth` | 各约 1G | 普通权重 + 优化器状态（仅断点续训用，推理不需要） |
| `metrics.json` | KB | 每 epoch 的 loss/AUC，**评估和台账都靠它** |

> 磁盘紧张的话，训练结束后删 `model_f*.pth`，每个模型省 6G。

### 4.4 （可选）训练 CheXFound 第二模型

CheXFound 需要第三方代码 + 4 个官方权重文件（Google Drive，国内需本地下载后人工
传到 `third_party/CheXFound_Model/`）。完整准备步骤和检查脚本见
[`README_TRAIN.md`](README_TRAIN.md) 第 2.1 节，就绪标志是
`python3 check_chexfound.py` 全绿：

```bash
python3 train.py --model chexfound
```

时间与 EVA-X 相当或略短。只有 EVA-X 也能先出第一个提交，不必等齐两个模型。

### 4.5 中断与恢复

SSH 断了训练就停了，重进 tmux 会话即可看到现场（`tmux attach -t train`）。
要从一半继续：

```bash
python3 train.py --model evax --resume runs/evax_xxx/model_f3.pth
```

---

## 5. 推理与生成提交文件（约 10~20 分钟）

```bash
cd ~/cxr/baseline
RUN=runs/evax_20261018_203000        # 换成你的实际目录名

# ① EVA-X：第 5、6 epoch + TTA（Kaggle 方案的标准配置）
python3 predict.py --model evax --split test --tta \
    --ckpt $RUN/ema_f5.pth $RUN/ema_f6.pth \
    --out $RUN/pred_test.npz

# ② 同时在 valid 上推一份（下载回 Mac 后本地评分用，文件只有几十 KB）
python3 predict.py --model evax --split valid --tta \
    --ckpt $RUN/ema_f5.pth $RUN/ema_f6.pth \
    --out $RUN/pred_valid.npz

# ③ 生成提交文件（双模型时把两个 npz 都写上）
python3 ensemble.py --split test \
    --inputs $RUN/pred_test.npz \
    --out submission.csv
```

**预期输出：**

```
图像数 2218 -> study 数 1414（聚合方式 mean）
已写入 submission.csv（14140 行，全 10 类）
提交文件自检通过（study 覆盖、行数、类别 id、概率范围均正确）
```

14140 = 1414 个 study × 10 类。自检不过就别提交，按提示修。

---

## 6. 下载回结果（约 5 分钟）

大权重（几 GB）留在 GPU 机上，只搬回小文件。在 **GPU 机**上先打包：

```bash
cd ~/cxr/baseline
tar czf ~/results_evax.tgz runs/*/metrics.json runs/*/pred_valid.npz submission.csv
```

在 **Mac** 上下载并解包：

```bash
cd /Users/mac/WorkBuddy/计挑赛/baseline
scp 用户名@10.20.30.40:~/results_evax.tgz .
mkdir -p runs && tar xzf results_evax.tgz
```

| 文件 | 大小 | 用途 |
|---|---|---|
| `submission.csv` | 几百 KB | **提交比赛用** |
| `metrics.json` | KB | 训练过程记录，进实验台账 |
| `pred_valid.npz` | 约 60 KB | Mac 本地重新评分/多模型对比 |

> 什么时候需要把 `best.pth`（345M）也下载回来：复赛要交模型时、或要在 Mac 上做
> 小规模验证时。平时不用搬。

---

## 7. 在 Mac 上评估效果

不需要 GPU、不需要图像数据，几秒到几分钟：

```bash
cd /Users/mac/WorkBuddy/计挑赛/baseline

# 单模型在 valid 上的正式评分（study 级 = 比赛评分单位）
python3 ensemble.py --split valid \
    --inputs runs/evax_xxx/pred_valid.npz --out /tmp/val_check.csv
```

输出：

```
[valid] study 级 Macro AUC: 0.8236
  每类 AUC:  1:0.6123  2:0.7455  3:0.7012  4:0.7823  5:0.8456
              6:0.8512  7:0.8301  8:0.8934  9:0.8812  10:0.9310
```

**怎么读这份分数：**

- **Macro AUC 是主指标**，十类等权——把最弱的类从 0.61 提到 0.70，比把最强的类
  从 0.89 提到 0.92 更划算；
- **valid 只有 859 个 study**，罕见类正样本个位数，AUC 波动大：两个模型差
  **0.005 以内不要下结论**，以每类 AUC 和提交回分综合判断；
- 想对比多个模型（如 EVA-X vs CheXFound vs 集成），把各自的 `pred_valid.npz`
  下载回来，`ensemble.py --inputs a.npz b.npz` 即可看到集成效果。

**每次实验必须记录（团队规则）：**

| 日期 | 实验名（runs 目录） | 模型/改动 | valid study AUC | 最弱 3 类 | 提交回分 | 备注 |
|---|---|---|---|---|---|---|
| 10-18 | evax_20261018_203000 | EVA-X 6ep 基线 | 0.8236 | 1/3/2 | — | 首跑 |

---

## 8. 提交到比赛平台

1. 登录比赛平台，上传 `submission.csv`（UTF-8，勿用 Excel 另存，否则编码/格式易坏）；
2. **每日限 2 次**——团队指定一人管理名额，不要各自为政；
3. 提交策略：第一次交 baseline 建立回分基线；之后只在 valid 有实质提升
   （Macro AUC +0.005 以上）时才交；
4. 提交口径若主办方确认「只交 >0.5 的类」，改 `config.py` 的
   `SUBMIT_ALL_CLASSES = False` 后重新生成即可。

---

## 9. 预期结果参考

### 时间预期（4× RTX 5090 32GB，EVA-X 单模型）

| 步骤 | 预期耗时 |
|---|---|
| 上传（代码+数据+权重 3.6G） | 5~15 分钟 |
| 装环境（venv + 约 2~4G 依赖） | 10~20 分钟 |
| 冒烟测试 | 1~2 分钟 |
| 训练 6 epoch（4 卡 DataParallel） | **20~40 分钟**（单卡 1~2 小时） |
| TTA 推理 + 生成提交 | 5~10 分钟 |
| 下载回 + 本地评估 | 5 分钟 |

### 分数预期（首跑基线，经验区间非承诺）

| 情况 | study 级 Macro AUC | 说明 |
|---|---|---|
| 正常 | **0.80 ~ 0.87** | EVA-X 医疗预训练 + 本数据 3 万张，首跑落在此区间算健康 |
| 偏低 | 0.75 ~ 0.80 | 可能正常（罕见类拖分），检查每类 AUC 是否普遍低 |
| **异常** | **< 0.75** | 大概率有 bug：预处理不匹配、归一化错、ckpt 没加载对。先查再调 |

分类别预期：常见类（8 肺部阴影 / 9 胸腔积液 / 10 无异常）通常 0.85~0.93；
罕见类（1 纵隔心影增大 / 2 气胸 / 3 肺部实变）通常 0.60~0.80 且逐 epoch 波动大
——这是任务性质决定的（正样本占比 3~4%），也是后续优化主战场。

### 训练过程预期

- 第 1 个 epoch 结束 AUC 通常已到 0.78+（预训练权重的功劳）；
- 之后每 epoch 涨 0.003~0.01，第 5、6 epoch 趋平——所以 Kaggle 方案平均最后两个
  epoch 的权重；
- loss 从约 0.15 降到 0.08~0.11 区间；如果 loss 不降或 AUC 不涨，见第 10 节。

---

## 10. 故障排查速查

| 症状 | 原因 / 解法 |
|---|---|
| `torch.cuda.is_available()` 为 False | ①装成了 CPU 版（常见于用了镜像源的 CPU wheel）或 ②CUDA 不匹配：按 3.4 重装；`nvidia-smi` 能跑但 torch 看不到 → 重装对应 CUDA 的 torch |
| torch 报 `CUDA driver version is insufficient` | 驱动太老带不动默认的 cu12x torch：改装 cu118 构建（3.4 表第二行） |
| 训练 OOM（CUDA out of memory） | `--batch-size 8`；仍不行把 `config.EVAX_IMG_SIZE` 降到 320 |
| `图像不存在: .../pic_xxx.jpg` | data600 没传全或路径不对；回第 2.3 节校验文件数 |
| `FileNotFoundError: ../cxr_first/train.csv` | 3 个标注 CSV 没传到服务器：按 2.2 第 ② 条传（只传 CSV，不传图片） |
| AUC 远低于预期（<0.75） | 依次查：ckpt 是否真的加载（看启动日志的「权重加载情况」）、预处理是否用了同一套、`--tta` 是否漏加 |
| TTA 推理报 pickle 错误 | 代码不是最新版；确认 `dataset.py` 里亮度缩放是模块级函数（2026-10-18 已修） |
| valid 推理的 npz 里 id 是数字 | 同上，旧版 Dataset 三件套 bug，更新代码 |
| HF 权重下载失败 | 已上传 weights/ 则不会触发；真需要下载时设 `CXR_HF_ENDPOINT=https://hf-mirror.com` |
| SSH 断线训练中断 | 用 `tmux` 跑；已存盘的 epoch 可用 `--resume runs/xxx/model_fN.pth` 续 |
| rsync 报 `link_stat ".../计挑赛/data600/." failed` 且开始传几 GB | `--partial` 与 `--exclude 'dir/'` 同用触发 rsync 2.6.9 bug：去掉 `--partial` 重跑（代码同步不需要它） |
| rsync 报 `mkdir ".../cxr/baseline" failed` | 远程父目录 `~/cxr` 不存在：先 `ssh 用户名@IP "mkdir -p ~/cxr"` |
| 提交文件被平台拒绝 | 用 Excel 打开过 → 编码坏了；重新在 Mac/Linux 上用本文命令生成，别用 Excel 碰它 |

---

## 附：常用命令速查

```bash
# Mac：上传代码（改代码后只跑这条；勿加 --partial，会让 exclude 失效）
rsync -av --exclude 'data600/' --exclude 'weights/' --exclude 'runs/' /Users/mac/WorkBuddy/计挑赛/baseline/ 用户名@GPU机IP:~/cxr/baseline/

# Mac：上传 3 个标注 CSV（仅首次；缺了服务器会报找不到 train.csv）
ssh 用户名@GPU机IP "mkdir -p ~/cxr/cxr_first"
rsync -av --partial /Users/mac/WorkBuddy/计挑赛/cxr_first/train.csv /Users/mac/WorkBuddy/计挑赛/cxr_first/valid.csv /Users/mac/WorkBuddy/计挑赛/cxr_first/test.csv 用户名@GPU机IP:~/cxr/cxr_first/

# GPU 机：训练 / 推理 / 提交
python3 train.py --model evax
python3 predict.py --model evax --split test --tta --ckpt runs/xxx/ema_f5.pth runs/xxx/ema_f6.pth --out runs/xxx/pred_test.npz
python3 ensemble.py --split test --inputs runs/xxx/pred_test.npz --out submission.csv

# Mac：下载小结果 + 本地评分
scp 用户名@GPU机IP:~/results_evax.tgz . && tar xzf results_evax.tgz
python3 ensemble.py --split valid --inputs runs/xxx/pred_valid.npz --out /tmp/val_check.csv
```
