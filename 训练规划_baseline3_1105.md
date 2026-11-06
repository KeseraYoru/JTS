# baseline3 训练规划（2026-11-05）

> 一句话：**换掉最弱的那个起点，同时把 baseline1 用两周换来的 8 条教训写进代码默认值。**
> 目标不是"再跑一遍"，而是让「换权重」这件事变成一次干净、可判定、可交付的实验。

---

## 一、为什么要开新基线

### 1.1 规则复查的结论（这是全部动机）

逐字重读《规则》第 1 条与《补充》二/五后确认：被禁的是**"预训练主数据集为医学"**的权重，
不是"所有强预训练"。规则原文点名允许：

> 以 **ImageNet、COCO、LAION** 等通用自然图像或**公开文本数据**预训练的模型视为合规

而我们两周里只用了其中**最弱的一档**：

| 类别 | 例子 | 主数据集 | baseline1 用过？ |
|---|---|---|---|
| ImageNet-1k 监督 | torchvision 权重 | IN-1k | ✅ 一直用 |
| **ImageNet-21k 监督** | ConvNeXt `*_22k` | IN-21k | ❌ |
| **网页级自监督** | **DINOv2**（LVD-142M，Apache-2.0） | 公开网页图 | ❌ |

这不是"钻空子"：规则补充二(一)写明"官方材料未把医学列为主数据集即视为合规"，
而 DINOv2 的 model card 明确写的是 LVD-142M 通用网页图像。

### 1.2 已有路线的判决（为什么必须换起点）

| 已关闭的杠杆 | 证据 |
|---|---|
| 分辨率 384→448 | +0.0004（A5 对照） |
| SSL 200 轮 | 50 轮即饱和，之后 +0.0012 |
| study 聚合 max/混合 | mean 最优，max −0.0023 |
| pos_weight 加码 | 峰值 +0.0001 |
| 同架构 5 折集成 | +0.0021（相关性太高） |
| 平衡采样 | −0.0086（同 batch 对照） |
| ASL 损失 | −0.0045（同 batch 对照） |
| data1024 换源 | +0.0024，但推理 +40ms，净亏 |

**能动的只剩两件事：换起点、加容量。** 本项目当前 valid 0.8027（平台 77.67），
复赛线 77.7 —— 差 0.03。换起点是唯一有望一次跨过去的动作。

---

## 二、baseline3 修掉了哪 8 个坑

| # | baseline1 的症状 | 根因 | baseline3 的做法 |
|---|---|---|---|
| 1 | batch8 上指标掉 0.0067，一度归咎于 batch | **LR 没跟着 batch 缩放** | `--lr auto` 默认按 `batch/32` 线性缩放；日志打印实际 lr |
| 2 | 15 轮 loss 一路降到 0.09，AUC 卡 0.58 | **EMA 写在 epoch 循环里**，几乎不更新 | `update()` 封装在训练 step 内；每轮打印累计步数，收尾核对 `总步数=轮数×每轮` |
| 3 | 加融合层后 ckpt 变 `unexpected keys`，predict 崩 | 推理端靠命令行猜结构 | ckpt 写全 `impl/arch/img_size/view_dim/pool`，推理端照抄 |
| 4 | 同一张原图被解码 N 遍，双模型 64ms | 每模型各读一次图 | `predict_one_multi` 共享一次读图+预处理（实测 42ms，预测逐位等价） |
| 5 | 冻结骨干报 `DataParallel has no attribute backbone` | 先包 DP 再调 set_trainable | 函数内部先解包，所有调用方安全 |
| 6 | 换骨干时 SSL 权重"同名不同形状"直接抛错打断训练 | `strict=False` 不容忍形状不符 | 先按「键名+形状」过滤再加载，打印匹配率，匹配率 <60% 显式告警 |
| 7 | 6 路 TTA 只 +0.001 却 6 倍成本 | TTA 种类选错 | 默认只 `base + hflip` 两种（获奖方案常用），可配 |
| 8 | "draft 解到 img_size 能省时间"是错的 | draft 只有 1/2、1/4、1/8 三档 | 代码注释写死这一点，省得下次再试 |

另外新增两件 baseline1 完全没有的东西：

- **ViewPosition 元数据**（4 维 one-hot）：同类赛事第二名靠"正位/侧位分开处理"拿银牌，
  我们 CSV 里一直有这个字段没用过。`--use-view` 开关，默认关。
- **Normal Gating 后处理**：类 10「无异常」与其余 9 类**严格互斥**（12649 行实证）。
  当模型确信"正常"时压低异常类概率 —— 确定性、逐样本、不拟合参数，规则允许。

---

## 三、三条路线与判据（先探针，再全量）

| 路线 | 命令里的 arch | 预期 | 判据（相对同批 IN-1k 探针） |
|---|---|---|---|
| A. DINOv2 ViT-B/14 | `dinov2_vitb14` | +0.010~0.020 | ep2 探针 **≥ +0.008** 才做全量 |
| B. ConvNeXt IN-22k | `convnext_base_in22k` | +0.004~0.008 | 同上 |
| C. 现状对照 | `convnext_base` | 基准 | — |

**纪律**：
1. 三个探针必须**同 batch、同卡、同种子、同轮数**（batch 不同则不可比，这是被坑过的地方）；
2. 探针只看 ep2（ep1 是冻结骨干的线性探针，只说明特征线性可分，不说明微调上限）；
3. 任何 <0.008 的差异一律判"无证据"，不许写"略有提升"。

---

## 四、执行步骤（照着敲）

### Step 0 · 上传与自检（10 分钟，不占 GPU）

```bash
# Mac
cd ~/WorkBuddy/计挑赛
scp -r baseline3 lcy@192.168.136.64:~/cxr/

# 服务器
cd ~/cxr/baseline3
bash download_weights.sh          # 约 1.1GB：DINOv2 330MB + ConvNeXt-22k 350MB
python -m py_compile *.py vendor/*.py tools/*.py && echo "语法 OK"
python tools/selftest_archs.py --arch dinov2_vitb14 --img-size 392 --weights weights/dinov2_vitb14_pretrain.pth
```

**必须看到**：
```
[权重] pos_embed 已插值: (1, 1370, 768) -> (1, 785, 768)
[权重] 匹配 173/174 键 | 跳过 0 | 缺失 1 | 多余 0
  前向 (2, 10) | 86.1M 参数 | fp16 164MB
```
- **匹配 173/174 是正常的**：唯一"缺失"的是 `mask_token`（官方 MAE 蒸馏用，我们主动丢弃），
  而 `pos_embed` 走插值路径，所以不在"匹配"计数里但确实被写入了；
- 若匹配率 <60% → 权重不是 ViT-B/14，别再往下走；
- 体积 164MB（fp16）→ 500MB 预算绰绰有余。

### Step 1 · 三个探针（约 1 小时，单卡）

```bash
cd ~/cxr/baseline3
export CXR_PREP_ROOT=~/cxr/baseline/data600

run_probe () {
  python -u train3.py --arch "$1" --img-size "$2" --batch-size 16 --lr 1e-4 \
    --layer-decay "$3" --probe \
    --train-csv runs/folds/fold_0_train.csv --val-csv runs/folds/fold_0_val.csv \
    --out "runs/probe_$4" 2>&1 | tee "runs/probe_$4.log"
}

run_probe convnext_base        448 1.00 base
run_probe convnext_base_in22k  448 1.00 in22k
run_probe dinov2_vitb14        392 0.75 dino
```

> 若 `runs/folds/` 不在新目录下，先 `cp -r ../baseline1/runs/folds runs/`。

**读结果**：
```bash
for t in base in22k dino; do
  python -c "
import json;d=json.load(open('runs/probe_$t/metrics.json'))
print('$t', ' '.join(f\"ep{r['epoch']}:{r['study_auc']:.4f}\" for r in d))"
done
```

判据表：

| dino − base | 动作 |
|---|---|
| ≥ +0.015 | 全面换主线，全量微调 + 与 ConvNeXt 双集成 |
| +0.008~+0.015 | 换，但保留 ConvNeXt 当集成成员 |
| < +0.008 | 不换；line B（IN-22k）若也无效，则回到现有双模型收尾 |

### Step 2 · 全量微调（胜者，约 1.5~2 小时/个）

```bash
# DINOv2（若胜出）
python -u train3.py --arch dinov2_vitb14 --img-size 392 --batch-size 16 \
  --lr 1e-4 --layer-decay 0.75 --epochs 12 \
  --out runs/full_dino 2>&1 | tee runs/full_dino.log

# 或 ConvNeXt IN-22k（若胜出）
python -u train3.py --arch convnext_base_in22k --img-size 448 --batch-size 16 \
  --lr 1e-4 --epochs 12 \
  --out runs/full_in22k 2>&1 | tee runs/full_in22k.log
```

**DINOv2 配方要点**（ViT 与 CNN 不同）：
- `--lr 1e-4`、`--layer-decay 0.75`（深层小步长）、`--warmup-epochs 1`
- 输入必须是 14 的倍数（392 = 28×28 token）
- 头学习率 ×10 保持；weight decay 0.05

**判据**：与 `full_base`（0.7997，含选择偏差）比，**≥ +0.008 才算赢**；
同时看类 1（当前 0.69）有没有被抬起来 —— 那是剩余分数的大头。

### Step 3 · 验证集预测 + 融合方式对比（5 分钟，零成本）

```bash
python -u predict3.py --split valid --ckpt runs/full_dino/best.pth   --out runs/pv_dino.npz
python -u predict3.py --split valid --ckpt runs/full_base/best.pth   --out runs/pv_conv.npz 2>/dev/null || \
python -u predict3.py --split valid --ckpt ../baseline1/runs/full_base/best.pth --out runs/pv_conv.npz
```

> 注意：`../baseline1/runs/full_base/best.pth` 是 baseline1 的旧 ckpt，**本基线读不了**
> （没有 arch 字段）。要用它做集成，先用 baseline1 的 predict 出 npz，再喂给 ensemble3。

**四种组合都测一遍**（纯 CPU，秒级）：prob 平均 / logit 平均 / 独立模型 / 加 gating。

### Step 4 · 出提交

```bash
python -u predict3.py --split test --ckpt runs/full_dino/best.pth --out runs/pt_dino.npz

# 单模型
python -u ensemble3.py --npz runs/pt_dino.npz --method mean \
  --out submission_dino.csv --save-meta runs/sub_dino_meta.json

# 双模型（跨架构）
python -u ensemble3.py --npz runs/pt_dino.npz runs/pt_conv.npz --method logit \
  --gate 1.0 --out submission_ens.csv --save-meta runs/sub_ens_meta.json
```

### Step 5 · 耗时与一致性核验（**这一步决定成绩是否有效**）

```bash
python -u cost3.py submission_dino.csv pictures_100 pictures_5 \
  --ckpt runs/full_dino/best.pth --meta runs/pt_dino_meta.json
```

看两行：`不一致 0 张`、`最终平均耗时 XX ms`。memorize：5090 上 DINOv2@392 预计 35~45ms，
折算 T4 约 65~75ms（余量 25%+）。

### Step 6 · 打包（终版才做）

```bash
# fp16 存权重（500MB 预算翻倍）
python - <<'EOF'
import torch
ck = torch.load("runs/full_dino/best.pth", map_location="cpu")
for k in ("model_state_dict", "ema_state_dict"):
    ck[k] = {n: (v.half() if torch.is_floating_point(v) else v) for n, v in ck[k].items()}
torch.save(ck, "package/model/dino_fp16.pth")
print("已存 fp16")
EOF
```

---

## 五、时间与算力预算（现实版：卡不是独占的）

| 步骤 | GPU 时间 | 说明 |
|---|---|---|
| 自检 + 下载 | 0 | 纯 CPU |
| 三个探针 | 3×20min = 1h | 单卡 batch16 |
| 全量微调（胜者） | 1.5~2h | 单卡 batch16 |
| 第二成员（可选） | 1.5h | 若要做双集成 |
| 预测 + 核验 | 10min | — |
| **合计** | **3~5h GPU** | 抢两个空闲窗口即可 |

**抢卡策略**：mtt 的 Pythia 训练长期占 4 卡，只能单卡。优先挑**显存最空的那张**
（`nvidia-smi --query-gpu=index,memory.used --format=csv`，选剩余 > 12GB 的），
`CUDA_VISIBLE_DEVICES=N`，batch16 只占约 8GB，不会挤到别人。

---

## 六、合规材料（A1~A6 怎么填）

| 项 | DINOv2 路线 | IN-22k 路线 |
|---|---|---|
| A1 模型名 | DINOv2 ViT-B/14（自监督） | ConvNeXt-Base（ImageNet-22k） |
| A2 权重链接 | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth` | `https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth` |
| A3 许可证 | **Apache-2.0**（GitHub 官方 release，**不要用 HF 上的 CC-BY-NC 镜像**） | MIT |
| A4 上游数据 | LVD-142M：公开网页图像自动筛选+LVD 去重，**主数据集非医学** | ImageNet-22k（通用自然图像） |
| A5 SHA-256 | `sha256sum weights/*.pth` | 同左 |
| A6 推理精度 | FP16（AMP） | FP16 |

**必须在申报里额外写一句**（防审核误判）：
> 本方案未使用任何以医学影像、医学文献或医学数据库为主数据集预训练的权重；
> 亦未使用任何外部医学数据（含无标签）进行预训练、续训或自监督。

---

## 七、风险与回退

| 风险 | 概率 | 应对 |
|---|---|---|
| DINOv2 权重与自实现结构对不上 | 中 | Step0 自检先验；匹配率 <60% 就停下查变体 |
| ViT 微调不收敛（LR 太大/太小） | 中 | 先 `--probe` 两轮看 ep2；不动则 LR 减半、warmup 加到 2 轮 |
| 392px 下 T4 超 100ms | 低 | 前向估算约 10ms，T4 约 70ms；真超了就降 364（26×26 token） |
| 探针显示无增益 | 中 | 直接放弃，回现有双模型 + T4 实测保 77.67 |
| 抢不到卡 | 高 | 单卡 batch16 也能跑，只是慢 2 倍；不要为此改配方 |

**回退链**（任何一步失败都不影响已有成绩）：
```
77.67（双架构集成，已 banked）
  └─ 若 DINOv2 有效 → 换主线，目标 78.5+
  └─ 若无效 → 用 large384 或现有双模型交付，全力做材料与撤销
```

---

## 八、实验台账模板（每跑一组填一行）

| 日期 | 实验名 | arch | img | batch | lr | 峰值 AUC | 峰值轮 | 类1 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 11-05 | probe_base | convnext_base | 448 | 16 | 1e-4 | | | | |
| 11-05 | probe_in22k | convnext_base_in22k | 448 | 16 | 1e-4 | | | | |
| 11-05 | probe_dino | dinov2_vitb14 | 392 | 16 | 1e-4 | | | | |

---

## 九、一句话复盘（写给未来的自己）

**这次换基线不是为了"再试一次"，而是因为规则复查发现我们把允许清单读窄了。**
如果一个方向连"起点"都选错了，再多的调参都是在错误的坐标系里努力。

---

## 十、已实测验证（2026-11-05 晚，本机 CPU）

新基线不是"写完就交"的：本机装了 CPU 版 torch（2.14.1 / torchvision 0.29.1），
把能验的全验了一遍。**验出 3 个真 bug，全部已修**。

### 10.1 架构自检（11 个全部通过）

| 架构 | 参数 | fp16 | 500MB |
|---|---|---|---|
| convnext_base | 88.6M | 169MB | ✓ |
| convnext_large | 197.8M | **377MB** | ✓ |
| efficientnet_v2_s / m | 21.5M / 54.2M | 41 / 103MB | ✓ |
| swin_t / swin_b | 28.3M / 87.8M | 54 / 167MB | ✓ |
| **convnext_base_in22k** | 87.6M | 167MB | ✓ |
| **dinov2_vitb14** | 86.1M | **164MB** | ✓ |
| dinov2_vits14 | 21.8M | 42MB | ✓ |

> 注意 convnext_large 的 fp16 是 377MB，**单独装得下但有 123MB 余量**；双模型集成时放不下。

### 10.2 权重加载（用合成权重验证，不依赖下载）

| 模型 | 结果 |
|---|---|
| DINOv2 ViT-B/14 | **173/174 键匹配**；`pos_embed` 1370→785 插值成功；`mask_token` 正确丢弃 |
| ConvNeXt IN-22k | **342/342 键 100% 覆盖**；官方 `downsample_layers.*`/`stages.i.j.*` 命名全部正确重映射；1000 类 head 正确丢弃 |

### 10.3 全链路冒烟（假数据 8/4/6 张）

```
train3 (1 轮) → predict3 (valid/test) → ensemble3 (mean / logit / gating) → cost3
```
- **EMA 更新 4 步 = 4 步/轮 ✓**（这条验的是"每 step 更新"，baseline1 的最大坑）
- **LR 自动缩放**：batch=2 → 3e-4×2/32 = 1.87e-05 ✓
- **ckpt 结构元信息可重建**：predict3 从 ckpt 读出 `vendor_dinov2/dinov2_vitb14/56px/view_dim=0`
- **训练评估与 predict 完全一致**：convnext+use-view 训练末轮 0.5000 → predict 复算 0.5000
- **cost3 一致性 0 张不一致**（含 gating=1.0 的复现）
- **双模型分支**：同 ckpt 两份 → 仍 0 张不一致；尺寸不一致时正确拒绝

### 10.4 修掉的 3 个 bug（都会静默影响精度）

| # | 症状 | 根因 | 影响 |
|---|---|---|---|
| 1 | DINOv2 前向报 `Dimension out of range` | 模型已返回 (B,C)，`forward_features` 仍按 4D 做 mean | 直接崩，容易发现 |
| 2 | `dinov2_vits14` 报 `KeyError` | `PRETRAIN_SOURCES` 只登记了 vitb14 | 直接崩 |
| 3 | **ConvNeXt 12 个 downsample 权重静默没加载** | `downsample_layers.{i}` 映射成 `stages.{i-1}`，偏了一位 | **不报错、指标莫名偏低**——最危险的一类 |

第 3 个是这轮验证最大的收获：它在服务器上只会表现为"换了 22k 权重怎么没涨"，
而不会给任何报错。现在有断言守着（342/342）。

### 10.5 本机复验命令（改完代码后先跑这个，再上传）

```bash
cd baseline3
PY=/Users/mac/.workbuddy/binaries/python/envs/default/bin/python

# ① 语法 + 跨模块导入
$PY -m py_compile *.py vendor/*.py tools/*.py

# ② 架构与体积
$PY tools/selftest_archs.py --img-size 392

# ③ 假数据全链路（需 /tmp/cxr_smoke，可用下面脚本重建）
CXR_DATA_ROOT=/tmp/cxr_smoke/cxr_first CXR_PREP_ROOT=/tmp/cxr_smoke/prep \
  $PY train3.py --arch dinov2_vitb14 --img-size 56 --batch-size 2 --epochs 1 \
  --workers 0 --pretrained none --out /tmp/cxr_smoke/run
```

> 本机没有原始大图时，`predict3 --split test` 会主动报错并提示用 `--src cache` 冒烟；
> **但正式提交必须用原图**，否则 cost3 一致性核验必然挂。

### 10.6 服务器证书过期（`certificate has expired`）怎么办

现象：`bash download_weights.sh` 报 `curl: (60) SSL certificate problem: certificate has expired`。

**这不是权重问题，也不是我们的代码问题**——是训练机的 ca-certificates 太旧，
或被校园网/代理做了 HTTPS 中间人。权重内容不受影响，最终靠 SHA-256 验收（申报 A5 用）。

三种处理，按速度排序：

| 方案 | 命令 | 说明 |
|---|---|---|
| ① 跳过证书校验（最快） | 脚本已内置：自动降级 `curl -k` 重试 | 手动等价命令见下 |
| ② 修 CA（一劳永逸） | `sudo apt-get update && sudo apt-get install -y ca-certificates` | 需要 sudo；有权限就做 |
| ③ 本机下载后上传 | Mac 上下载 → `scp` 到 `baseline3/weights/` | 服务器网络完全不通时用 |

手动等价命令（方案①）：

```bash
cd ~/cxr/baseline3/weights
curl -k -L --retry 3 -o dinov2_vitb14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
curl -k -L --retry 3 -o convnext_base_22k_224.pth \
  https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
ls -lh                      # 期望约 330MB / 420MB
sha256sum *.pth             # 记下来，打包申报要填
```

> 已修：`archs.resolve_pretrained_path` 之前遇到"路径不存在"会误报成
> "`--pretrained` 只能是 auto/none/本地路径"，现在会明确说"权重文件不存在"并给出上面三种解法。
