# MacBook 使用指南（baseline/ 配套文档）

> 本指南只讲**在这台 Mac 上**怎么用这套代码。完整方案说明见 `README.md`，
> 本文件是它的 Mac 专属补充。

## 0. 这台 Mac 的角色定位

| 能做（推荐在 Mac 做） | 不能做（要去 GPU 机） |
|---|---|
| 数据预处理：46GB 大图 → 600² 小图（纯 CPU，约 20~40 分钟） | 完整训练（Mac 没有 NVIDIA GPU，CPU 跑一个 epoch 要几十小时） |
| 代码 review、改 `config.py` 里的超参 | —— |
| CPU 小样本冒烟测试（`--limit`，验证 torch 链路能跑通） | —— |
| 提交文件格式演练（`ensemble.py` 生成 + 自检） | —— |

> 关于 MPS（Apple 芯片的 GPU 加速）：EVA-X 的旋转位置编码等算子未必在 MPS 上
> 实现完整，且没有 AMP，收益有限。**不建议在 Mac 上折腾训练**，把时间花在
> 预处理 + 冒烟验证上，把整块训练时间留给机房 GPU。

## 1. 环境准备

**本机已经准备好的环境**（开箱即用）：

```bash
/Users/mac/.workbuddy/binaries/python/envs/default/bin/python3 --version
# numpy / pillow / tqdm 已装好，足够跑 preprocess.py 和逻辑测试
```

**要跑 torch 相关的东西（冒烟测试）才需要装全量依赖**：

```bash
cd baseline
/Users/mac/.workbuddy/binaries/python/envs/default/bin/python3 -m pip install -r requirements.txt
# Mac 上 pip 会自动装 CPU 版 torch；timm / albumentations / opencv 都支持 Apple 芯片
```

> 想完全独立一份环境也可以：`python3 -m venv .venv && source .venv/bin/activate`。
> 注意别往系统 Python 里装（`pip install` 尽量在 venv 里执行）。

## 2. 路径配置（默认就是为本机配好的，不用改）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `DATA_ROOT` | `../cxr_first` | 46GB 原始数据就在这台 Mac 上，相对路径直接可用 |
| `PREP_ROOT` | `./data600` | 预处理输出（约 3.5GB） |
| `RUNS_DIR` | `./runs` | 训练输出（Mac 上只会放冒烟测试的产物） |
| `WEIGHTS_DIR` | `./weights` | EVA-X 预训练权重（344MB，首次用时自动下载） |

换机器时用环境变量覆盖即可，例如：
`CXR_PREP_ROOT=/data/cxr600 python train.py --model evax`

## 3. 第一步：预处理（在 Mac 上做正事）

```bash
cd baseline
caffeinate -i /Users/mac/.workbuddy/binaries/python/envs/default/bin/python3 preprocess.py
```

- `caffeinate -i` 防止 Mac 睡着睡着把任务挂起（合盖前必加）；
- 预计 **20~40 分钟**（取决于核数；Air 系列无风扇机型可以 `--workers 4` 温和一点）；
- 输出约 **3.5GB**，开跑前先 `df -h .` 确认剩余空间 > 6GB；
- **中断了没关系**：已生成的文件会自动跳过，重跑同一条命令会接着干；
- `--split train` 可只处理训练集；`--limit 100` 可先试 100 张。

## 4. 把数据传到 GPU 训练机

原图 46GB **不用传**（训练只读 600² 小图）。上传的完整命令、远程目录准备、
断线续传和注意事项见 **[`README_GPU.md`](README_GPU.md) 第 2 节**，这里是摘要：

```bash
# ① 先建远程目录（rsync 只创建最后一级，父目录缺失会 mkdir failed）
ssh 用户名@GPU机IP "mkdir -p ~/cxr"

# ② 代码（勿加 --partial：rsync 2.6.9 上它会让 --exclude 失效，实测踩坑）
rsync -av --exclude 'data600/' --exclude 'weights/' --exclude 'runs/' baseline/ 用户名@GPU机IP:~/cxr/baseline/

# ③ 数据 2.5G（无 exclude，--partial 安全，断了重跑续传）
rsync -av --partial --progress baseline/data600/ 用户名@GPU机IP:~/cxr/baseline/data600/

# ④ 预训练权重 1.1G（同上）
rsync -av --partial --progress baseline/weights/ 用户名@GPU机IP:~/cxr/baseline/weights/
```

到 GPU 机后：`pip install -r requirements.txt` → 按 `README_GPU.md` 第 3 节起
验证环境、第 4 节开始训练。如果 GPU 机上没有原数据，也可以直接在 GPU 机上跑
`preprocess.py`（Linux 上更快），把 `CXR_DATA_ROOT` 指过去就行。

## 5. （可选）Mac 上的 CPU 冒烟测试

目的：在上 GPU 机之前，先把「模型能构建 → 权重能下载 → ckpt 能存能读 →
推理能出概率」这条链在 Mac 上验一遍，把 GPU 时间留给纯等待。

```bash
# ① 先确保装了全量依赖（见第 1 节）
# ② 小样本训练 1 个 epoch（首次会自动下载 344MB 权重，几分钟）
/Users/mac/.workbuddy/binaries/python/envs/default/bin/python3 train.py \
    --model evax --limit 64 --epochs 1 --batch-size 4 --out runs/mac_smoke

# ③ 用小 ckpt 推理几张（验证 TTA / ckpt 读取，几分钟）
/Users/mac/.workbuddy/binaries/python/envs/default/bin/python3 predict.py \
    --model evax --split valid --limit 8 --tta \
    --ckpt runs/mac_smoke/ema_f1.pth --out /tmp/smoke.npz

# ④ 再走一遍提交生成（验证聚合 + 长表格式 + 自检，秒级）
/Users/mac/.workbuddy/binaries/python/envs/default/bin/python3 ensemble.py \
    --split valid --limit 8 --inputs /tmp/smoke.npz --out /tmp/smoke_sub.csv
```

三条命令都能跑完 = torch 链路完全没问题（2026-10-17 已在本机实测通过：
predict 6 种 TTA 全过、npz 字段正确、ensemble 生成 40 行提交且自检通过）。
注意：

- 冒烟测试的 AUC **没有参考意义**（64 张图什么也学不出来），只看能不能跑通；
- 跑完把 `runs/mac_smoke/` 删掉（约 1.7GB，其中 model_f1.pth 含优化器状态约 1GB），
  别和正式实验混在一起；
- 权重下载失败时设 `CXR_HF_ENDPOINT=https://hf-mirror.com` 再试（国内镜像）。

## 6. 常见问题

| 症状 | 原因 / 解法 |
|---|---|
| `df -h .` 剩余空间不足 | 46GB 原图 + 3.5GB 输出，先清出 6GB 以上 |
| 预处理跑到一半变慢/像卡住 | Mac 睡着了 → 加 `caffeinate -i`；或在「设置-电池-显示器」里插电时禁止睡眠 |
| `pip install` 报权限错 | 别往系统 Python 装，用上面的 venv 路径或 `--user` |
| `uname -m` 显示 `x86_64` | 当前 shell 跑在 Rosetta 下，能跑但慢；换 arm64 原生的 python3 |
| HuggingFace 下载超时 | 设 `CXR_HF_ENDPOINT=https://hf-mirror.com`，或让队友下载后放到 `weights/` |
| 想跳过预处理直接用原图训练 | 把 `CXR_PREP_ROOT=../cxr_first`，能跑但每个 epoch 慢 5~10 倍，只建议调试用 |

## 7. 代码 review 建议顺序

1. `README.md` —— 先看总览和「与 Kaggle 原版的差异」表；
2. `config.py` —— 所有旋钮都在这，5 分钟看完；
3. `common.py` —— 数据怎么从 CSV 变成标签、概率怎么变提交文件（逻辑最集中）；
4. `train.py` —— 训练主循环（每个 epoch 干什么、存哪些文件）；
5. `models/evax.py` —— 技术含量最高的一段（从 Kaggle 逐行移植，已加注释）。
