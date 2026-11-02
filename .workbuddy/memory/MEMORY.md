# 项目长期备忘：胸部X光多标签分类竞赛

## 用户画像
- 人工智能专业大二学生，**机器学习零基础**（人生第一次训练模型）。沟通时必须从最底层概念讲起、用生活比喻，每个术语当场解释，避免只抛结论。
- 明确偏好：在用户说「开始」之前不要写任何代码文件，先对齐思路。

## 赛题不变量（长期有效）
- 任务：胸部X光片多标签分类，10 类，输出每类概率。
- 类别 id：1 纵隔心影增大 / 2 气胸 / 3 肺部实变 / 4 肺炎 / 5 肺水肿 / 6 心影扩大 /
  7 肺不张 / 8 肺部阴影 / 9 胸腔积液 / 10 无异常发现。
- 数据：train 18513 检查/30344 图、val 859/1422、test 1414/2218（test 无标注）。实际路径
  `cxr_first/{train,valid,test}/subject_x/study_y/pic_z.jpg`（Dicom_id=pic_ 开头）。
- **Predict_class 格式**：单列、分号分隔类别 id，如 `7;9`、`3;5;9`；空串=无异常需按数据确认（实测无空值）。
- **标签是 study 级**：18513 个多图 study 中标签不一致数为 0 → 同 study 所有图共享标签，按图训练即可。
- **类别 10（无异常）与其余 9 类互斥（数据实证）**：含 10 的行 = 纯 10 的行 = 12649。
- 类别正样本数（按图行，n=30344）：10:12649 / 8:6703 / 9:6637 / 7:5649 / 6:5590 / 5:3220 /
  4:2293 / 3:1265 / 2:1238 / 1:894（最罕见类仅 2.95%，不均衡约 14 倍）。
- 图像：8bit 灰度 JPEG，尺寸 1409~4280px 无固定；train 46GB / valid 2.1GB / test 3.4GB。
- split 无泄露：subject 与 study 均无交叉；study id 区间 train 0~19999 / valid 20000~20937 / test 20938~22437。
- ViewPosition 有 1202 行空值；CSV 中**没有** Support Devices 列（题面提到但未提供）。
- 关键字段：Subject_id、Study_id、Predict_class、ViewPosition、PatientOrientation、Rows、Columns。
- 评估：主指标 Macro AUC，辅指标 Macro-F1，主指标相同时才比较辅指标。
- 提交：submission.csv，UTF-8 长表，Subject_id/Study_id/Predict_class/Probability，每日限 2 次。
- 硬约束：T4 16GB、单张 ≤100ms、模型压缩前 ≤500MB、PyTorch 2.10 + torchvision 0.25、
  CUDA 12.8、仅用主办方数据（允许增强，不允许外部标注数据）。

## 已在方案中确立的技术原则
- Macro AUC 与阈值无关 → 不做阈值调优；优化重心是每类的排序质量。
- 十类等权 → 罕见类对总分影响与常见类相同，需按 per-class AUC 定位薄弱类。
- 官方 val 仅 859 条，罕见类正样本过少 → 必须自建 patient 级验证集做主判据。
- 提交概率必须保留 6 位小数，避免并列拉低 AUC。
- 训练与推理的图像尺寸、预处理必须严格一致；复赛容器离线，权重需随镜像打包。
- **粒度错配（重要）**：模型输入是单张图，但评分与提交单位是 Study_id（一次检查可能含 1~多张图）→ 必须做 study 级聚合（mean / max / 按体位加权），聚合方式是可调超参，需在自建验证集上定。
- 图片无固定尺寸 → resize + 归一化方案须在训练/推理两端写死一致（320² 起步，T4 余量充足可试更大）。
- 官方点名三大难点：样本不均衡、标签共现、类间相似性高 → 分别对应类别加权采样/focal loss、标签相关性建模、更高分辨率与更长训练。
- 元数据暂未利用，后期可用：ViewPosition（正/侧位影响病灶可见性）、PatientOrientation、Support Devices（1/0/-1/None，重症信号）。

## 高优先级待澄清（决定方案主线）
1. **提交口径**：官方写"默认以 0.5 为阈值，多个类别可多行输出"，但 Macro AUC 需完整概率矩阵 → 暂定交全 10 类（1414×10 = 14140 行），确认后再裁剪；受每日 2 次提交限制，不可拿名额试错。
2. ~~**ImageNet 预训练权重是否允许**~~ → **已解决**：用户已确认 EVA-X/CheXFound 医疗预训练权重可用（2026-10-17），迁移学习主线成立，直接按 Kaggle 方案走。
3. **标签口径**：类别 10（无异常）与另外 9 类是否互斥 → **数据实证互斥**（含10=纯10=12649）；"未提及/不确定"如何映射默认视为 0（实测 CSV 无空标签，此问暂不影响）。

## Kaggle 参考方案（Grand-X-Ray-Slam，Div A 亚军）要点
- 仓库：`Grand-X-Ray-Slam-competition/`（README + div A/B notebook + documentation.pdf）。方案：600² 预缩放 + EVA-X(448) + CheXFound(512) 双模型集成，Focal+BCE、EMA、layer-wise LR、TTA、ckpt 平均。
- **可借鉴**：预缩放大图、WeightedFocalBCE（effective-number pos_weight, β=0.9999）、EMA(0.9997)、TTA（原图+hflip+多尺度+亮度，共 6 种取平均）、checkpoint 平均（第 5+6 epoch）、EVA-X 6 epoch、CheXFound 冻结 backbone 只训 GLoRIA 头。
- **⚠ 外部权重合规性（2026-10-27 书面规则推翻了此前的口头确认）**：`计挑赛/规则.md` 第 1 条 +
  `规则补充.md` 二 明确「**预训练主数据集为医学影像/医学文献/医学数据库的模型不合规**」，点名禁止
  「以病理、放射、超声等医学影像为主数据集」的权重；补充一(二)写明**赛题页/FAQ/群内答疑/口头答复
  不构成规则例外，与书面不一致的以书面为准** → **EVA-X（MIMIC-CXR）与 CheXFound（CheXpert）
  在复赛（区域赛/省赛）均不合规**（除非能拿到书面/正式公告的例外许可）。
- 复赛硬约束（规则第 7/8 条）：**权重等解压后 ≤500MB**（CheXFound ViT-L 1.2G 单独就超、
  双模型集成 ~1.5G 更超 → 只能交单个小模型）；**单张端到端 ≤100ms**（100~150ms 可整改复测一次
  且只许优化推理实现不许换权重、>150ms 直接不予认定）；**必须交付 cost.py**（导入并调用与
  predict.py 相同的推理/加载逻辑，100 张 batch=1 串行跑 3 轮，输出须与 submission.csv
  样本级完全一致，|Δp|≤1e-3 才算浮点波动）。
- **合规可用手段（补充四明确允许）**：以合规权重（ImageNet 等）初始化，用官方训练集
  （30344 张）做**自监督预训练/续训**后再有监督微调 —— 不违规，且能补医学域适应。
- 明确禁止：外部医学数据（含无标签）预训练/续训/伪标签；**知识蒸馏的教师模型也必须非医学**
  （不能用 EVA-X/CheXFound 当教师，学生初始化权重同样须非医学）；测试集不得用于任何统计量/
  阈值/checkpoint 选择。
- 明确允许：TTA、多模型集成、**患者平滑**（同一 study 多图确定性聚合如取均值，性质同 TTA/集成，
  我们当前的 study mean 聚合即属此类，合规）。
- 验证集（第 3 条）：可并入训练，但并入后不得再用于早停/ckpt 选择/阈值/超参调整（须并入前定死），
  且需申报（未申报属轻度违规）。
- 提交包（附录 A）：`队伍名_最终提交材料.zip` = code/ + model/ + pretrain_model/ + licenses/ +
  cost.py + 模型核验材料与数据申明清单.xlsx + submission.csv；另需申报 A1~A6（模型名、公网直链、
  许可证全文、上游数据说明、SHA-256、推理精度）。
- 评分口径（补充十）：主指标后台 0~1 保留 4 位，页面展示百分制保留 2 位（0.8108→81.08），
  仅当展示值完全相同才启用副指标，再相同按提交时间较早者优先。容器只保证 torch+torchvision。
- **已落地代码**：`baseline/`（12 个 py + README + requirements + setup_chexfound.sh）。Kaggle notebook → 模块化脚本；`7;9`→10 维多热；路径从 CSV 三列拼；提交按 Study_id 聚合（多图 mean）成长表；增强用 albumentations，AUC 用 numpy 自写（sklearn 可选）。
- 跑法：preprocess.py → train.py --model evax/chexfound → predict.py --tta --ckpt ema_f5+ema_f6 → ensemble.py → submission.csv。已过 py_compile + common.py 逻辑测试 + 20 张预处理实测；torch 模型部分已在 Mac CPU 小样本全链路验证（train→predict→ensemble 冒烟通过），GPU 正式训练待用户上机首跑。
- **五份 README**（README.md 顶部有导航）：README.md（总览+Kaggle 差异对照+文件清单）、README_TRAIN.md（日常操作手册 Step0~6）、README_MAC.md（Mac 端）、README_GPU.md（首次上机：上传/环境/排查）、**README_CHANGES.md（改动注释集：每个坑的症状→根因→修法→验证 + 文件功能 + 数据流 + 待办）**、**TUTORIAL_多标签分类.md（教学向：多标签概念/AUC/搭流水线/技巧why/踩坑实录/提分）**。文档引用文件与命令参数已自检一致。
- **预期基准（4×5090）**：EVA-X 6 epoch 约 16min（288s/epoch 是 CheXFound）；首跑 study 级 Macro AUC 0.80~0.87 正常，<0.75 先查 bug 别调优。
- **首版提交结果（2026-10-22）**：平台回分 **0.7936**，本地 valid 等权集成 0.8240 / CheXFound 单模 0.8263。落差归因（已实测分析）：
  ① **valid 太小**：859 study，罕见类正样本仅 24/37/46 个，单类 AUC 标准误 ±0.056/0.046/0.041，Macro 平均后约 **±0.01** → 小验证集上小数点后第三位无意义（已核对 valid 与 train 类别分布几乎一致，排除分布偏移）。
  ② **概率整体偏低**：提交文件各类概率均值仅 0.18~0.42；类 10（真实占比 41.7%）仅 9% 的预测 ≥0.5。AUC 是排序指标不受整体偏移影响，**但若平台按题面「0.5 阈值」处理就会显著拉低**——这是最该验证的一条（诊断命令见 README_CHANGES.md 末尾）。
  ③ test 本身更难（无标注，无法验证）。

## 合规重写版 `baseline1/`（2026-10-30 起的主线）
- **动机**：书面规则禁止医学预训练权重（EVA-X/CheXFound 不合规），用户决定换模型、走「效果更强」的合规路线。
- **技术方案**：torchvision **ImageNet-1k** 骨干（默认 `convnext_base`，87.6M 参数 / fp32 334MB，
  500MB 内最强单模型之一）→ **官方训练集 SimCLR 自监督预训练**（规则补充四允许）→ 12 epoch 有监督微调
  （差分学习率：骨干 3e-4 / 头 3e-3；bias&norm 不做 weight decay；cosine 预热；EMA 0.9997；AMP；DataParallel）。
- **依赖极简**：只用 torch + torchvision + numpy + pillow + tqdm（去掉 timm/albumentations/cv2/fvcore），
  与复赛容器一致；数据变换全部改用 `torchvision.transforms.v2`。
- **合规交付物（已实现并本地验证）**：
  - `cost.py`：100 张 batch=1 串行 ×3 轮计时 + 与 submission.csv 一致性核验（|Δp|≤1e-3、
    二值判定与 top-1 排序），复用 `inference.py`（与 predict.py 同一份代码，规则第 7 条硬要求）；
  - `check_compliance.py`：体积(≤500MB)/SHA-256/许可证/推理确定性/A1~A6 申报表；
  - `make_package.py`：一键生成 `队伍名_最终提交材料.zip`（code/model/pretrain_model/licenses/
    cost.py/清单.xlsx/submission.csv），xlsx 含 A1~A6 + B1~B7 申明；
  - `licenses/BSD-3-Clause_torchvision.txt`（合规材料，须进 git）。
- **复用**：`data600` 预处理产物直接复用（600 ≥ 所有输入尺寸）；losses/ema/preprocess/common/
  ensemble/scan_weights 从 baseline 沿用（与模型无关）。
- **已验证（Mac CPU）**：16 个 py 语法通过；convnext_base 构建/前向/优化器 4 参数组；Dataset 三件套；
  predict 4 张 → npz；ensemble → 30 行提交自检通过；cost.py 一致性 **0 张不一致**；打包结构符合附录 A；
  合规自检通过；确定性 |Δp| = 0。
- **⚠ EMA 必须「每训练 step」更新（2026-10-30 踩过，代价 4 次 15 轮训练全废）**：
  `train_one_epoch(..., ema=ema)` 里调 `ema.update(model)`；若放在 epoch 循环里只调 15 次，
  EMA 15 轮后只混入 0.45% 新权重（实测移动量 0.24%）→ **每轮打印的 AUC 其实是「训练前的模型」**，
  症状是 loss 0.21→0.09 稳定下降、study AUC 卡在 0.58 不动、不同 seed 首轮 0.49/0.58 乱跳。
  `ema.py` 已加 warmup（`decay=min(base, n/(n+1))`）+ in-place foreach 加速；
  `train.py` 每轮打印 `EMA 更新 N 步`，必须 = 轮数×每轮步数。**修复前产出的 ckpt 全部作废**。
- **待 GPU 机上跑**：SSL 20 epoch（约 1~2h）→ 微调 12 epoch → 推理 → cost.py 实测
  （T4 上 384px 单张预计 60~90ms，含 TTA 需实测）→ 打包。

## 赛程时间线
- 数据开放：距用户首次提问时还有约 3 天。
- 提交截止：数据开放后约 1 个月结束（有效竞赛期约 30 天）。
- 推论：首次训练模型者学习成本高，赛前 3 天窗口必须用于环境搭建与基础训练能力练习，而非空想模型结构。

## 训练机实况（2026-10-19 确认）
- **学校服务器：4× NVIDIA RTX 5090 32GB（共 128G 显存）**，驱动 580.126.09 / CUDA 13.0，
  主机名 enine（lcy@192.168.136.64），venv 在 ~/cxr/venv，torch 已装好 CUDA 13.0 构建可用。
- 题面的 T4 16GB 是**复赛推理验证**规格；训练用这 4 张 5090，EVA-X 6 epoch 约 20~40 分钟（4 卡 DataParallel）。
- 已修复多卡 DataParallel 下的优化器构建 bug（解包 model.module）；batch 默认 32。

## 团队协作（多人赛）
- 赛制为多人赛：用户 + 2 名队友，共 3 人。队友技术水平未知，待了解后再细化分工。
- 分工原则（已给用户）：
  - 阶段 A（数据开放第 1~3 天）**三人不分工**，共同打通同一条流水线（环境 / EDA / baseline），避免代码分叉、新手掉队。
  - 阶段 B（第 1~2 周）起按「低风险但必须有人做」给用户派活：实验台账、提交格式校验、TTA；两名队友分认主干/分辨率、损失函数与类别不均衡等消融方向。
  - 阶段 C（第 3~4 周）交叉验证与模型集成需合并预测，必须由一人统一调度。
- 三条硬规则：① 单一 git 仓库，各自只改自己的模块，不私下 fork；② 每日 2 次提交额度指定一人管理，禁止各自为政浪费名额；③ 所有实验必须落实验台账（日期 / 改动 / 每类 AUC / 提交分）。
- 复赛需提交模型供验证 → 最终模型须可复现并附文档。
