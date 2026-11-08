# 项目长期备忘：胸部X光多标签分类竞赛（已压缩，细节见 memory/日期.md）

## 协作
- 用户：AI 专业大二，ML 零基础（首次训模型）。中文、要生活比喻、术语当场解释。
- **改代码必须同时给出可直接上传/粘贴的完整文件 + 上传命令 + 自检命令**。

## 赛题不变量
- 10 类多标签，输出每类概率；1 纵隔心影增大 … 10 无异常。
- train 18513 study/30344 图，val 859/1422，test 1414/2218（无标注）。路径 `cxr_first/{split}/subject_x/study_y/pic_z.jpg`。
- 标签 study 级（同 study 所有图共享）；**类别 10 与其余 9 类互斥**（含10=纯10=12649）。
- 正样本（按图）：10:12649 8:6703 9:6637 7:5649 6:5590 5:3220 4:2293 3:1265 2:1238 1:894。
- 提交：长表 Subject_id/Study_id/Predict_class/Probability，UTF-8，每日 2 次，取历史最高。
- 主指标 Macro AUC（阈值无关，只比排序）；辅 Macro-F1。
- 交付：权重 ≤500MB、单张端到端 ≤100ms（100~150 可整改一次，>150 不认定）、必交 cost.py（与 predict.py 同一推理代码，100 张 batch=1 ×3 轮，|Δp|≤1e-3 且 0 张不一致）、容器只保证 torch+torchvision。

## 合规红线（以书面规则为准，口头/FAQ/群答无效）
- **禁止**：医学影像为主数据集的预训练权重（EVA-X、CheXFound 不合规）；外部医学数据（含无标签）训练/伪标签；蒸馏教师也须非医学；测试集不得用于任何选择。
- **允许**：ImageNet/COCO/LAION 权重；用官方训练集做自监督预训练；TTA；多模型集成；study 级多图均值聚合。
- 验证集可并入训练，但并入后不得再用于早停/选 ckpt/调参，且须申报。

## 核心结论（最重要）
- **官方 valid 对平台分无预测力**（5 个提交点相关≈0）：platform ≈ valid×100 − 3.02±0.30。
- 唯一裁判是平台提交；策略=造多个满足交付约束的候选，用额度撞分布上尾。
- 已证伪：SSL>50ep、448 训练、6 路 TTA、DINOv2、平衡采样、ASL、batch8、data1024、9 个小模型、知识蒸馏。
- 有效：SSL +0.0258、IN-22k 起点 +0.0050、跨架构集成 +0.003（平台 +0.79）、全量数据 +0.0047。

## 现状
- 平台最高 **77.67**（baseline1: full_base+effv2m，277MB，5090 上 39ms → T4 约 100ms，**待实测**）。
- 待交：in22k30+swinb30（valid 0.8112，334MB，估 T4 92ms）> in22k30+effv2s30（0.8102，208MB）。
- 目标 ≥77.7 复赛线，78.5 一等奖线。

## 代码与环境
- 服务器 enine `lcy@192.168.136.64`，4×RTX5090，venv `~/cxr/venv`，用 tmux + `python -u` 跑。
- `baseline1/`：合规主线（torchvision + 自建 SSL + convnext_base），产出 77.67；**无 --aug/--mixup**。
- `baseline3/`：8 架构注册表（含 IN-22k、DINOv2），有 aug/mixup/layer-decay，新实验一律走这里。
- 预处理产物 `data600` 复用；**EMA 必须每 step 更新**（放 epoch 层会让全部 ckpt 报废）。
- ConvNeXt timm 权重 remap 有 off-by-one 坑（`stages.{i}` → `stages.{i-1}`），已修，须核对权重数 342/342。
