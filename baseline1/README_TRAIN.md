# README_TRAIN.md —— 最高分路线（500MB / 100ms 约束下）

> 这份手册只讲一件事：**在规则允许、且最终只能交一个 334MB 模型的前提下，怎么把分数做到最高。**  
> 每一步都写了「做什么 / 预期看到什么 / 为什么这一步值这点时间」。

---

## 0. 先想清楚：为什么是这套方案

| 常规提分手段                        | 在本赛题能不能用 | 原因                                                           |
| ----------------------------- | -------- | ------------------------------------------------------------ |
| 多模型集成（3 个模型）                  | ❌        | 1 个 convnext_base ≈334MB，3 个就 1GB，**超 500MB 上限**             |
| **多 seed + 权重平均（model soup）** | ✅        | 平均后**仍是 1 个模型、1 份体积**，却吃到多模型的一部分收益                           |
| TTA（一张图多变换取平均）                | ✅        | 不增加体积，规则明确允许                                                 |
| ImageNet 骨干 + 官方数据自监督         | ✅        | 补医学域适应，规则补充四明确允许                                             |
| 用满数据（train+valid）             | ✅        | 多 4.7% 数据，规则第 3 条允许（选择须在并入前定死）                               |
| 多模型真集成（两个**小**模型）             | ⚠️ 可选    | convnext_small + efficientnet_v2_m ≈418MB，能过，但单个更弱，见「可选方案 B」 |

**一句话**：主线 = **SSL 域适应 → 3 个 seed 各训一次 → 权重平均合成 1 个模型 → TTA → 提交**。

### 总览

| 步骤     | 做什么                             | 4×5090 耗时     | 产出                            |
| ------ | ------------------------------- | ------------- | ----------------------------- |
| Step 0 | venv + 数据检查 + 1 分钟冒烟            | 10 分钟         | —                             |
| Step 1 | 自监督预训练（官方 train）                | 1.5~2 小时      | `runs/ssl/ssl_backbone.pth`   |
| Step 2 | 3 个 seed 微调（12~15 epoch）        | 3~4 小时        | 3 个 `runs/s{seed}/ema_fN.pth` |
| Step 3 | 权重平均（soup）                      | 10 秒          | `runs/soup.pth`（334MB）        |
| Step 4 | valid 上对比 soup vs 单模型           | 20 分钟         | 决定用哪个                         |
| Step 5 | 全量重训（train+valid，3 seed 再 soup） | 3~4 小时（可选但推荐） | `runs/final_soup.pth`         |
| Step 6 | 推理（TTA）                         | 15~20 分钟      | `pred_test.npz`               |
| Step 7 | 出提交 + 本地评分                      | 1 分钟          | `submission.csv`              |
| Step 8 | cost.py 核验                      | 10 分钟         | 耗时 + 一致性结论                    |
| Step 9 | 合规自检 + 打包                       | 5 分钟          | `队伍名_最终提交材料.zip`              |
| Step 10 | **下载回本地**（在 Mac 上：打包 → scp → 校验） | 2 分钟          | `results/` 里拿到 submission.csv  |

**最短可跑通路径**（时间紧）：Step 0 → 2（只训 1 个 seed）→ 6 → 7 → 8 → 10，约 2 小时，先拿到一个合规基线分。

---

## Step 0. 准备

### 0.1 虚拟环境（首次必做）

**为什么要建 venv**：服务器通常多人共用，直接 `pip install` 到系统 Python 会搅乱别人的依赖；  
venv 是独立环境，坏了删掉重建只要五分钟，数据/代码/权重都在 venv 外面，删环境不会丢东西。

```bash
# ① 建环境（已存在就跳过；沿用 ~/cxr/venv）
python3 -m venv ~/cxr/venv

# ② 激活：提示符前出现 (venv) 才算成功
source ~/cxr/venv/bin/activate

# ③ 装依赖（清华镜像；本方案只需 torch/torchvision/numpy/pillow/tqdm）
cd ~/cxr/baseline1
pip install -U pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ④ 验证：版本 + CUDA
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# 期望：2.x.x  0.x.x  True
```

**常见问题**

| 现象                            | 处理                                                                                                                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `No module named venv`        | `sudo apt install -y python3-venv`；或改用 conda                                                                                                                                                        |
| `cuda.is_available()` 为 False | 装成了 CPU 版：按服务器 CUDA 版本重装（如 `--index-url https://download.pytorch.org/whl/cu121`）                                                                                                                    |
| 服务器不能上外网                      | 有网机器上 `pip download -r requirements.txt -d wheels/ --platform manylinux2014_x86_64 --python-version 310 --only-binary=:all:`，拷过去后 `pip install --no-index --find-links wheels/ -r requirements.txt` |

**日常使用**：每次新开 SSH 窗口都要 `source ~/cxr/venv/bin/activate`；`which python` 确认指向 venv。

### 0.2 数据检查

```bash
ls ../cxr_first/*.csv                             # 期望 train.csv valid.csv test.csv
find ../baseline/data600 -name "*.jpg" | wc -l    # 期望 33984（复用 baseline 的预处理产物）
find . -maxdepth 1 -name "*.py" | wc -l           # 期望 18 个
```

> `data600` 不用重跑：600 比所有候选骨干的输入尺寸（384/512）都大，从这里再 resize 即可。

### 0.3 预置 ImageNet 权重（**别省这一步**）

首次运行会自动从 `download.pytorch.org` 下载 338MB 的 ImageNet 权重。国内直连速度不可控  
（实测有 42 kB/s 的情况，等于卡 2 小时）。所以**先从 Mac 传上去**，LAN 内只要十几秒：

```bash
# ① Mac 上：建远程目录 + 传（沿用之前传 EVA-X 权重的做法）
ssh lcy@192.168.136.64 "mkdir -p ~/.cache/torch/hub/checkpoints"
rsync -av --partial ~/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth \
    lcy@192.168.136.64:~/.cache/torch/hub/checkpoints/

# ② 服务器上：校验（大小 338M，哈希必须一致）
ls -lh ~/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth
sha256sum ~/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth
# 期望：6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd
```

Mac 上没有这份文件的话，任选其一：

- 在 Mac 上跑一次 `python -c "from models import build_model; build_model('convnext_base')"`  
  （Mac 的直连速度通常比服务器好），再把缓存文件传过去；
- 或者有代理时在服务器上 `export https_proxy=...` 后重试。

**换骨干时要换对应文件**（文件名里的 8 位十六进制就是 SHA-256 前缀，校验用）：

| 骨干                | 权重文件                             | 大小    |
| ----------------- | -------------------------------- | ----- |
| convnext_base（默认） | `convnext_base-6075fbad.pth`     | 338MB |
| convnext_small    | `convnext_small-0c510722.pth`    | 192MB |
| efficientnet_v2_m | `efficientnet_v2_m-dc08266a.pth` | 208MB |
| swin_b            | `swin_b-68c6b09e.pth`            | 335MB |
| vit_b_16          | `vit_b_16-c867db91.pth`          | 330MB |

下载地址统一是 `https://download.pytorch.org/models/<上面那个文件名>`。

> 自监督（Step 1）和微调（Step 2）用的是**同一个** convnext_base 权重，传一次就够。

### 0.4 一分钟冒烟（权重就位后再跑）

```bash
rm -rf runs/preflight
python train.py --limit 64 --epochs 1 --batch-size 8 --out runs/preflight
```

**预期**：不再出现 `Downloading:` 一行；打印「训练集 64 张 | batch 8 | 每轮 8 步 × 8 张 ≈ 64 张次」，  
约 1 分钟内跑完 1 个 epoch 并保存 `ema_f1.pth`。AUC 数值无意义（只是冒烟），能存盘就删掉 `runs/preflight`。

> 如果看到 `Downloading:` 说明权重没放对位置（目录是 `~/.cache/torch/hub/checkpoints/`，  
> 文件名必须完全一致），Ctrl+C 停下来按 0.3 传一次，别等它慢慢下。

### 0.5 长任务防掉线（**必看**）

**为什么**：SSH 一断，前台进程就被杀，训练直接终止。tmux 让进程挂在服务器的会话里，  
断线只影响你的终端，不影响训练。

```bash
atmux new -s train            # 新建会话
python train.py --ssl ... --out runs/s48        # 在里面跑
# 想离开：先按 Ctrl+B，松开后再按 D（detach），进程继续跑
tmux attach -t train         # 回来时接上
tmux ls                      # 看有哪些会话
```

不想用 tmux 可以用 nohup：

```bash
nohup python train.py --epochs 15 --ssl runs/ssl/ssl_backbone.pth --seed 48 \
    --out runs/s48 > runs_s48.log 2>&1 &
tail -f runs_s48.log
```

**掉线之后，按顺序做三件事**

```bash
tmux ls                              # ① 会话还在吗
nvidia-smi                           # ② 训练进程还在跑吗（有 python 占着 GPU 就是没死）
tail -n 20 runs/<run>/metrics.json   # ③ 到第几个 epoch 了
```

| 情况                      | 处理                                   |
| ----------------------- | ------------------------------------ |
| tmux 会话还在、GPU 还在跑       | `tmux attach -t train` 接回去，**什么都别动** |
| 进程没了，但目录里有 `ema_fN.pth` | 用下面的「续训」命令接着跑                        |
| 进程没了，什么都没有              | 说明第一轮就掉了，直接从头重跑                      |

**续训**

```bash
python train.py --ssl runs/ssl/ssl_backbone.pth --epochs 15 --seed 48 \
    --resume runs/s48/ema_f7.pth --out runs/s48
```

- 会打印「已完成 7 轮，学习率曲线快进 X 步」——**学习率接着原来的余弦曲线走**，  
  不会从最大值重新爬升（这个坑之前在 EVA-X 上踩过）；
- 优化器动量（AdamW 的 m/v）没存盘，所以属于「热重启」：通常一两个 epoch 内恢复，  
  分数可能出现 ±0.003 的波动，属正常；
- **掉线发生在很靠前（比如第 2 轮）时，直接从头重跑更干净**——15 轮也就 60~75 分钟。

**防患于未然**：在 Mac/本机的 `~/.ssh/config` 里加心跳，避免闲置被断开：

```
Host enine
    HostName 192.168.136.64
    User lcy
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

之后直接 `ssh enine` 即可，长时间无输出也不会断。

> 进度最多损失**一个 epoch**：`train.py` 每轮结束都会写 `ema_fN.pth` 和 `metrics.json`。

---

## Step 1. 自监督预训练（1.5~2 小时，强烈建议）

**为什么值这个时间**：ImageNet 是自然照片，X 光是灰度、低对比、纹理完全不同。  
先用官方 3 万张图做无标签自监督（SimCLR），骨干会先学会「看懂这类图」，  
之后再微调通常比直接微调高 1~3 个点。**这是 500MB 约束下唯一能补域适应的手段**（换更大的医学预训练模型不允许）。

**合规性**：规则补充四明确允许「用官方训练集做自监督预训练/续训」；脚本只用 `train` split，  
不碰 valid/test，不引入任何外部数据。**须在申报表 B4 行注明。**

```bash
tmux new -s ssl
python ssl_pretrain.py --epochs 20
```

**预期**

```
设备 cuda | 骨干 convnext_base | batch 128 | epochs 20
数据范围：仅官方 train split（规则补充四允许的自监督范围）
SSL Epoch 1/20 | loss 4.82 | 205s
...
自监督权重已保存: runs/ssl/ssl_backbone.pth
```

- loss 从 ~4.8（≈ln(2B−1)）缓慢降到 3.x 属正常，不要求收敛到很低；
- 4 张 5090 上 20 epoch 约 1.5~2 小时；显存不够就 `--batch-size 96`。

---

## Step 2. 三个 seed 各训一次（3~4 小时）

**为什么训三次**：单次训练的分数有 ±0.01 的随机波动（初始化、数据顺序、dropout），  
三次取平均能把这份波动消掉一大半——Step 3 的权重平均就是靠这三份模型。

```bash
tmux new -s train
for s in 48 49 50; do
  python train.py --ssl runs/ssl/ssl_backbone.pth --epochs 15 --seed $s --out runs/s$s
done
```

> **实测校准（2026-10-31，convnext_base @384）**：15 轮里峰值出现在 **epoch 12**
> （study AUC 0.7995），之后 ep13/14/15 连续下降到 0.7991 → 0.7977 → 0.7958（累计 −0.0037）。
> 原因不是「轮数多」而是**学习率还没退到 0**：15 轮方案下 ep12 的骨干 LR 还有 5.7e-5，
> 继续推着模型越过最优点。若第 12 轮时 LR 已退火到 8.6e-6（即 `--epochs 12`）通常能再稳一点。
> **所以：再跑时用 `--epochs 12`**；本轮三个 seed 保持 15 轮不变（三个 seed 配方一致，
> 权重平均才干净），各自取 `best.pth`（自动停在峰值轮）做 soup 即可。

**预期（每个 seed）**

```
设备: cuda | AMP: True | 骨干: convnext_base | batch: 32 | epochs: 15 | seed: 48
训练集 30344 张 | batch 32 | 每轮 949 步 × 32 张 ≈ 30368 张次（每轮都完整过一遍训练集）
验证集 1422 张（859 个检查），每轮结束后评估
[Train] epoch 1: 100%|████| 949/949 [04:30<00:00, 3.5it/s, loss=0.2140, lr=3.00e-04]
Epoch 1/15 | loss 0.2140 | 图像级 AUC 0.7812 | study 级 AUC 0.7790 | 280s | EMA 更新 949 步
  每类 AUC:  1:0.6800  2:0.8000 ... 10:0.8500
...
Epoch 15/15 | ... | study 级 AUC 0.8xxx | 280s | EMA 更新 14235 步
```

> **`EMA 更新 N 步` 这一项要盯住**：它必须等于「已训轮数 × 每轮步数」。
> 如果第 15 轮显示的是 `EMA 更新 15 步`，说明 EMA 被放进了 epoch 循环——
> EMA 会几乎冻结在初始权重，你看到的 AUC 是「训练前的模型」（2026-10-30 踩过，
> AUC 卡在 0.58 不动而 loss 一路降）。正确写法是**每个训练 step 调一次 `ema.update()`**。

**健康判据**

| 指标           | 正常                       | 异常                                    |
| ------------ | ------------------------ | ------------------------------------- |
| study AUC    | 第 1 轮 ≥0.75，逐轮升，末轮 ≥0.80 | <0.70 先查 bug（归一化/标签/路径），别调参           |
| loss         | 0.21 → 0.10 附近           | 卡 0.5+：学习率 ×3；震荡：×0.3                 |
| 三个 seed 末轮分数 | 相互差 ≤0.01                | 差得大 → 训练不稳定，检查 `--seed` 是否真的不同、数据是否打乱 |

> 每个 seed 约 60~80 分钟。时间紧就先只跑 seed 48，后面再补。

---

## Step 3. 权重平均（10 秒）

```bash
python soup.py --ckpt runs/s48/ema_f15.pth runs/s49/ema_f15.pth runs/s50/ema_f15.pth \
    --out runs/soup.pth
```

**预期**

```
合并 3 个模型：
  - runs/s48/ema_f15.pth（epoch 15, valid 0.80xx）
  ...
✓ 已保存 runs/soup.pth（334.2 MB，仍是单模型体积）
```

**注意**：`ema_f15.pth` 要换成你实际的轮数（15 epoch → `ema_f15.pth`）。  
权重平均的前提是「同一个骨干、同一套参数名」——脚本会自动校验，不一致会直接报错。

---

## Step 4. 决定用 soup 还是单模型（20 分钟）

```bash
python soup.py --ckpt runs/s48/best.pth runs/s49/best.pth runs/s50/best.pth \
    --out runs/soup.pth --eval-split valid
```

> **⚠ 实测校准（2026-10-31）**：三个 seed 同起点、同配方时，**soup 的收益约等于「三个模型的平均」，
> 而不是「高于平均」**。实测：s48 0.7994 / s49 0.7984 / s50 0.7964（均值 0.7981），
> soup = **0.7977**——比最好的单模型低 0.0017，和均值基本重合。
>
> 原因：权重平均能消掉「种子带来的抖动」，但不会产生「错误互补」的增益——后者只有**预测级集成**
> （把三个模型的概率平均）才有，而那需要同时加载 3 份权重（3×334MB=1002MB）→ 超 500MB 上限。
>
> **结论**：soup 是「同体积下更稳」的选择，但不是提分手段。真要吃到集成的分，
> 用**知识蒸馏**把三模型集成蒸到一个模型里（规则允许，教师非医学即可）。

> **用 `best.pth` 而不是 `ema_f15.pth`**：`best.pth` 是「study AUC 最高的那一轮」的 EMA 权重，
> 自动避开了 peak 之后的轻微过拟合。先查三个 seed 各自的峰值轮再决定喂哪些文件：
>
> ```bash
> python -c "
> import json,glob
> for f in sorted(glob.glob('runs/s*/metrics.json')):
>     d=json.load(open(f)); b=max(d,key=lambda r:r['study_auc'])
>     print(f.split('/')[1], '峰值 ep', b['epoch'], round(b['study_auc'],4), '| 末轮', round(d[-1]['study_auc'],4))
> "
> ```

**决策规则（写死，别纠结）**

| 情况                       | 用哪个                                 |
| ------------------------ | ----------------------------------- |
| soup ≥ 最好的单模型            | ✅ 用 soup                            |
| soup 比最好的单模型低 **<0.003** | ✅ 用 soup（分数在噪声内，但 soup 方差更小、换测试集更稳） |
| soup 低 **>0.005**        | ❌ 用最好的单模型（说明三个模型跑得太开，权重不在同一盆地）      |

> 859 个检查的 AUC 噪声约 ±0.01，**0.003 以内的差异不要当真**。

---

## Step 5. 全量重训（train+valid，3~4 小时，推荐）

**为什么**：正式比赛要用上全部有标注数据（多 4.7% 图像、228 个患者）。  
但规则第 3 条要求：**验证集一经并入，就不能再用于早停/选 ckpt/调参**。  
所以顺序必须是：**Step 2~4 先把配方和轮数全部定死 → 再并入 → 直接取最后一轮**。

```bash
for s in 48 49 50; do
  python train.py --ssl runs/ssl/ssl_backbone.pth --epochs 15 --seed $s \
      --data train+valid --out runs/final_s$s
done

python soup.py --ckpt runs/final_s48/ema_f15.pth runs/final_s49/ema_f15.pth \
    runs/final_s50/ema_f15.pth --out runs/final_soup.pth
```

**注意**

- 这一步**不要再挑 epoch**（没有干净的验证集了），统一取**最后一个 epoch 的 EMA 权重**；
- 训练日志会提示「⚠ 已启用 train+valid」，这是正常的合规提醒；
- 时间不够就只训 1 个 seed，直接用它的 `ema_f15.pth`。

---

## Step 6. 推理（15~20 分钟）

```bash
CKPT=runs/final_soup.pth          # 或 runs/soup.pth / 某个单模型
python predict.py --split valid --ckpt $CKPT --tta --out runs/pred_valid.npz
python predict.py --split test  --ckpt $CKPT --tta --out runs/pred_test.npz
```

**预期**

```
已加载: runs/final_soup.pth | 骨干 convnext_base | 384px
[test] TTA1/6 ... TTA6/6
已保存 runs/pred_test.npz | probs.shape=(2218, 10)
推理参数已记录: runs/pred_test_meta.json      ← cost.py 要用它对齐参数
```

> `_meta.json` 会记录 ckpt / tta / 尺寸，**Step 8 必须用它**，否则 cost.py 与提交文件可能不一致。

---

## Step 7. 出提交文件 + 本地评分

```bash
# valid 自评（不上传，先看分数）
python ensemble.py --split valid --inputs runs/pred_valid.npz --out /tmp/val.csv

# 正式提交（14140 行，自检通过）
python ensemble.py --split test --inputs runs/pred_test.npz --out submission.csv
```

**预期**：`图像数 2218 -> study 数 1414`、`已写入 submission.csv（14140 行，全 10 类）`、  
`提交文件自检通过`。

---

## Step 8. cost.py 核验（必须）

```bash
python cost.py submission.csv pictures_100 pictures_5 --meta runs/pred_test_meta.json
```

**预期**

```
设备: cuda | 推理精度: fp16 (AMP 自动混合精度) | 骨干: convnext_base | 输入 384px | TTA: True | 聚合: mean
预热图 5 张 | 计时图 100 张
  第 1/3 轮完成，本轮均值 62.3 ms
  ...
[CONCLUSION] 结论：100 张样本、batch=1 串行、连续 3 轮；...最终平均耗时 62 ms，未超过上限；
预测结果与 submission.csv 比对 100 张，不一致 0 张；推理精度 fp16 (AMP 自动混合精度)。
```

**两条硬线**：平均 ≤100ms、不一致 0 张。

| 问题       | 处理                                                  |
| -------- | --------------------------------------------------- |
| 超 100ms  | 先看是不是 TTA 拖的（关 TTA 复测）；再考虑降分辨率 / 半精度。复测只许优化实现，不许换权重 |
| 有 1 张不一致 | 检查 `--meta` 是否指向生成提交时的那个文件；或 cost.py 与提交用了不同 ckpt   |

---

## Step 9. 合规自检 + 打包

```bash
python check_compliance.py --ckpt runs/final_soup.pth
python make_package.py --ckpt runs/final_soup.pth --submission submission.csv --team 你的队名
```

**预期**：体积 ≤500MB、许可证齐备、确定性 `|Δp|=0`、A1~A6 逐条打印；然后生成  
`package/你的队名_最终提交材料.zip`（code/ model/ pretrain_model/ licenses/ cost.py 清单.xlsx submission.csv）。

**打包后必须人工做两件事**：

1. 打开 `模型核验材料与数据申明清单.xlsx`，把 **B3（验证集是否并入）**、**B4（是否做自监督）**  
   按实际情况改成「是/否」并写明细节（不要留占位）。
2. 确认包里的 `submission.csv` 是**历史最高分**那一份。

---

## Step 10. 下载回本地（在 Mac 上做）

### 10.1 只下小文件，绝不下权重

先记住这条：**服务器上 `runs/` 里每个 epoch 一个 ckpt，每个约 334MB**。三个 seed × 15 轮 ≈ **15GB**，
传回本地没有意义（本地也不训练）。要下的是「能看懂结果 + 能提交」的那几个小文件。

| 要下 | 大小 | 用途 |
|---|---|---|
| `submission.csv` | ~530 KB | 直接上传比赛平台 |
| `runs/*/metrics.json` | 几 KB | 画训练曲线、对比各个 seed |
| `runs/pred_test.npz` | ~90 KB | 本地复算 / 存档 |
| `runs/pv_*.npz` | ~60 KB | valid 预测，用来本地复算 AUC |
| `runs/*_meta.json` | 1 KB | 记录 TTA / 输入尺寸等推理参数 |
| `package/你的队名_最终提交材料.zip` | ~320 MB | **只在最后交付时传**，日常迭代不用 |
| ❌ `runs/**/*.pth` | 334 MB × N | **不下**，留在服务器 |

### 10.2 服务器上打包

```bash
cd ~/cxr/baseline1

# 先把「存在的」小文件收集起来，避免 tar 因个别文件缺失而整体失败
FILES="submission.csv"
for f in runs/s*/metrics.json runs/pred_test.npz runs/pv_*.npz runs/*_meta.json; do
  [ -f "$f" ] && FILES="$FILES $f"
done

tar czf results_1031.tgz $FILES
ls -lh results_1031.tgz        # 期望 1 MB 以内
```

> ⚠ **绝对不要** `tar czf x.tgz runs/`——那会把 15GB 权重全打进去（3 seed × 15 轮 × 334MB）。
> 判断标准很简单：**压缩包超过 10MB，就是打错东西了**，停下来检查。

### 10.3 Mac 上拉回来（从 Mac 发起，别在服务器上推）

```bash
cd /Users/mac/WorkBuddy/计挑赛
mkdir -p results
scp lcy@192.168.136.64:~/cxr/baseline1/results_1031.tgz results/
tar xzf results/results_1031.tgz -C results/
```

`results/` 已在根目录 `.gitignore` 里，不会污染 git 仓库。

### 10.4 落地校验（三步，别省）

```bash
cd /Users/mac/WorkBuddy/计挑赛/results

head -3 submission.csv                    # 表头 + 前两行，看格式对不对
wc -l < submission.csv                    # 期望 14140（1414 检查 × 10 类）
ls -lh *.npz runs/*.npz 2>/dev/null       # 每个 npz 约 60~90 KB
```

本地复算一遍 AUC，确认和服务器上一致（这是验证「文件没传坏」最直接的办法）：

```bash
cd /Users/mac/WorkBuddy/计挑赛/baseline1

CXR_DATA_ROOT=/Users/mac/WorkBuddy/计挑赛/cxr_first \
python ensemble.py --split valid --inputs ../results/runs/pv_notta.npz --out /tmp/check.csv
```

打印的 `study 级 Macro AUC` 应该和服务器上那一模一样（解包后 npz 的路径是
`results/runs/pv_notta.npz`；如果你只打了 `pred_test.npz`，这条可以跳过）。

### 10.5 真要下权重时（只在最后交付或换机器时）

**用 rsync 断点续传，不要用 scp**（334MB 断了要重来）：

```bash
rsync -av --partial --progress \
  lcy@192.168.136.64:~/cxr/baseline1/runs/final_soup.pth \
  /Users/mac/WorkBuddy/计挑赛/baseline1/weights/
```

> 注意 macOS 自带 rsync 2.6.9 上 `--partial` 与 `--exclude 'dir/'` 同用有 bug，
> 但这里**没有 exclude，是安全的**。

### 10.6 两个必须避开的坑

1. **提交文件绝对不要用 Excel 打开**——Excel 会重写编码和数字格式，平台会直接拒收。
   要看内容用 `head`、`less` 或 VS Code。
2. **scp 的通配符要加单引号**：`scp 'host:~/runs/*.npz' .`，让通配符在服务器端展开；
   不加引号的话是本地 shell 先展开，会报 `No such file`。

### 10.7 顺手清服务器磁盘

```bash
# ① 先看谁在占地方
du -sh ~/cxr/baseline1/runs/* | sort -h | tail

# ② 临时实验整个删（probe 的结论已经拿到，权重没用了）
rm -rf ~/cxr/baseline1/runs/probeA ~/cxr/baseline1/runs/probeB

# ③ 每个 seed 只留 best.pth，删掉所有中间 epoch
#    best.pth 存的就是「峰值轮」的 EMA 权重，中间那些 334MB 的文件没有保留价值
cd ~/cxr/baseline1/runs/s48 && rm -v ema_f*.pth
cd ~/cxr/baseline1/runs/s49 && rm -v ema_f*.pth
cd ~/cxr/baseline1/runs/s50 && rm -v ema_f*.pth

# ④ 确认只剩精华
ls -lh ~/cxr/baseline1/runs/s48/
```

> 删之前确认 **`runs/soup.pth` 已经生成**——soup 是从 best.pth 算出来的，
> soup 有了之后中间 ckpt 就彻底没用了。三个 seed 清完大约能省 **13GB**。

---

## 可选方案 A：患者级 K 折（要系统性调参时才用）

用途不是"提分"，而是**让选择更可靠**：官方 valid 只有 228 个患者，做超参对比时噪声太大；  
K 折里每折验证有 **2211 个患者 / 6069 张图**，结论可信得多。

```bash
python make_folds.py --k 5 --seed 48 --out runs/folds
# 输出会打印每折的图数/患者数/各类正样本数，并校验「无患者跨折」

# 逐折训练（每折 1 个模型，约 5×60 分钟）
for i in 0 1 2 3 4; do
  python train.py --ssl runs/ssl/ssl_backbone.pth --epochs 15 \
      --train-csv runs/folds/fold_${i}_train.csv \
      --val-csv   runs/folds/fold_${i}_val.csv --out runs/cv_f$i
done
```

5 折分数取平均 = 该配方的可靠估计；比较两个配方时看这个数，而不是看 859 个检查的 valid。

> ⚠ K 折产出的 5 个模型**不能同时提交**（5×334MB 超限）。它们的作用是"选配方/估分数"，  
> 最终仍然要回到 Step 5 的全量重训。

---

## 可选方案 B：两个小模型真集成（另一条路）

如果想让「集成」真正发生（而不是权重平均），必须用**小模型**才装得下：

```bash
# convnext_small ≈200MB + efficientnet_v2_m ≈217MB = 417MB（<500MB）
python train.py --backbone convnext_small --ssl runs/ssl/ssl_backbone_small.pth --epochs 15 --out runs/small
python train.py --backbone efficientnet_v2_m --epochs 15 --out runs/effv2m
# 各自推理 → 扫权重 → 集成
python scan_weights.py --inputs runs/small/pred_valid.npz runs/effv2m/pred_valid.npz
python ensemble.py --split test --inputs runs/small/pred_test.npz runs/effv2m/pred_test.npz \
    --weights 0.6 0.4 --out submission.csv
```

注意：这个方案要额外做 SSL（`--backbone convnext_small`），且两个模型都弱于 convnext_base。  
**先跑主线，有富余时间再试这条。**

---

## 常见问题

| 症状                                          | 处理                                                                                    |
| ------------------------------------------- | ------------------------------------------------------------------------------------- |
| `FileNotFoundError: ../cxr_first/train.csv` | 3 个 CSV 没传（见 Step 0.2）                                                                |
| 权重下载卡住                                      | 手动下载后放 `~/.cache/torch/hub/checkpoints/`，文件名需与官方一致                                    |
| OOM                                         | `--batch-size 16`；仍不行换 `convnext_small`                                               |
| 训练 AUC 停滞                                   | 确认日志有 `[SSL] 已加载 ...`；没有说明 `--ssl` 路径错了                                               |
| soup 比单模型差很多                                | 说明几个模型跑散了，直接改用最好的单模型（见 Step 4 决策表）                                                    |
| 提交包超 500MB                                  | 只算推理时加载的权重；换小骨干或去掉集成                                                                  |
| cost.py 超 100ms                             | `--meta` 指向不带 TTA 的推理、或降分辨率；复测不许换权重                                                   |
| **SSH 掉线，训练没了**                             | 先 `tmux ls` + `nvidia-smi` 确认死没死；死了就用 `--resume runs/<run>/ema_fN.pth` 续训（见 Step 0.5） |
| **续训后分数掉了一截**                               | 优化器动量重置导致的热重启，通常 1~2 轮恢复；掉线太早（<3 轮）建议直接重跑                                             |
| **想中途看进度**                                  | `tail -f runs/<run>/metrics.json`；或 `tmux attach -t train`                            |
| **loss 一路降，AUC 却卡在 0.5x 不动**                | 十有八九是 EMA 只在每个 epoch 更新（`ema.py` 的 per-step 修复）。查日志里 `EMA 更新 N 步` 是否等于 `轮数×每轮步数`；**2026-10-30 前的 ckpt 全部作废，必须重训** |
