# README_COMPLIANCE.md —— 规则逐条对照与申报指引

> 依据：`计挑赛/规则.md`（11 条 + 附录 A）与 `计挑赛/规则补充.md`（2026-10-27）。
> 这份文档回答两个问题：**我们哪里不合规了（已改）**、**提交前还要交什么、怎么填**。

---

## 0. 一句话结论

`baseline/` 使用的 EVA-X（MIMIC-CXR）与 CheXFound（CheXpert）属于「以医学影像为主数据集
预训练」的权重，被规则第 1 条与补充二明确禁止；且补充一(二)规定**群内答疑/口头答复不构成
例外许可**，此前「主办方说可以用」的口头确认不作数。
`baseline1/` 已全部换成 torchvision 的 **ImageNet-1k** 权重，合规且不依赖任何医学预训练。

---

## 1. 逐条对照

| 条款 | 要求要点 | 我们的做法 | 落在哪 |
|---|---|---|---|
| **第 1 条** 预训练模型 | 只用非医学开源权重；非 NC、非 gated；须提交名称/直链/许可证/上游数据/哈希 | 全部 ImageNet-1k（torchvision，BSD-3-Clause，公网直链可下）；`config.PRETRAIN_SOURCES` 登记 | `models.py` / `config.py` / `licenses/` |
| **第 2 条** 训练数据 | 只用官方数据微调；禁止外部医学数据（含无标签） | 训练只读 `cxr_first/train`（可选 valid）；无任何外部数据 | `train.py` / `ssl_pretrain.py` |
| **第 3 条** 验证集 | 可并入训练，但并入后不得再用于早停/选 ckpt/调阈值；须申报 | `--data train+valid` 前先把所有选择定死；清单 B3 行注明 | `train.py` |
| **第 4 条** 增强 | 训练可用任意增强；推理关闭随机算子 | 训练随机增强；推理只有确定性变换（TTA 6 种固定） | `dataset.py` |
| **第 5 条** 测试集 | 不得用于预训练/微调/伪标签；不得算统计量/阈值/选 ckpt | 测试集只在最终推理时读；无任何拟合 | `predict.py` / `inference.py` |
| **第 6 条** 推理 | 固定参数前向；BN/LN 统计量冻结；禁止梯度/测试时训练；TTA 与集成允许 | `model.eval()` + `@torch.no_grad()`；TTA/集成为确定性平均 | `inference.py` |
| **第 7 条** 耗时 | 100 张 batch=1 串行 ×3 轮；≤100ms；须交付 cost.py 且与 predict.py 同逻辑 | `cost.py` import `inference.py`（与 `predict.py` 同一份）；含一致性核验 | `cost.py` / `inference.py` |
| **第 8 条** 体积 | 推理加载的权重解压后 ≤500MB | 默认 ConvNeXt-Base fp32 ≈354MB；集成需小模型组合 | `check_compliance.py` |
| **第 9 条** 撤销 | 全队仅一次撤销机会 | 提交前先本地验证，不轻易用撤销 | 流程约定 |
| **第 10 条** 可复现 | 结果可复现 + 随交模型材料 | 固定种子；`*_meta.json` 记录推理参数；`requirements.txt` 极简 | `train.py` / `predict.py` |
| **第 11 条** 违规处罚 | 未申报验证集、未注明精度属轻度违规 | 清单 B3/B4/A6 必填 | `make_package.py` |
| **附录 A** 提交包 | 队伍名_最终提交材料.zip（code/model/pretrain_model/licenses/cost.py/清单.xlsx/submission.csv） | `make_package.py` 一键生成 | `make_package.py` |
| **补充二** 预训练范围 | 医学影像/文献/数据库预训练不合规；点名 RETFound 等 | 已彻底不用医学预训练权重 | 全局 |
| **补充三** 蒸馏 | 教师模型也必须非医学；学生初始化同样须非医学 | 未使用任何蒸馏 | — |
| **补充四** 官方数据自监督 | **允许**用官方训练集自监督/续训 | `ssl_pretrain.py`（SimCLR，仅 train split），清单 B4 申报 | `ssl_pretrain.py` |
| **补充七** 患者平滑 | 同一 study 多图确定性聚合（均值/投票）属允许 | `aggregate_by_study` 默认 mean | `common.py` |
| **补充九** 超时处置 | 100~150ms 可整改复测一次（只优化实现，不换权重）；>150ms 不认定 | 训练时就把耗时纳入验收 | `cost.py` |
| **补充十** 评分口径 | 后台 0~1 保留 4 位，页面百分制 2 位；同分才比副指标 | 无需额外动作 | — |

---

## 2. 申报材料怎么填（A1~A6 / B1~B7）

`make_package.py` 会自动生成 `模型核验材料与数据申明清单.xlsx`，其中：

| 编号 | 内容 | 自动填的内容 | 你需要人工确认 |
|---|---|---|---|
| A1 | 模型名称 | torchvision ConvNeXt-Base（ImageNet-1k 预训练） | 若换了骨干，改这里 |
| A2 | 权重公开链接 | `download.pytorch.org/models/convnext_base-6075fbad.pth` | 确认链接公网可直接访问、无需登录 |
| A3 | 许可证全文 | BSD-3-Clause，见 `licenses/` | 确认不是 NC 类、不是 gated |
| A4 | 上游训练数据说明 | ImageNet-1k（通用自然图像）+ model card 链接 | 备好 model card / 论文 |
| A5 | 预训练权重哈希 | 本地缓存存在时自动算 SHA-256 | 没有就下载后计算补上 |
| A6 | 推理精度 | FP16（AMP） | 与实际一致 |
| B1 | 最终权重文件与哈希 | 文件名、大小、SHA-256 | — |
| B2 | 推理时加载权重合计 | 自动算，需 ≤500MB | 集成时确认 |
| B3 | **验证集是否并入训练** | 占位「见训练记录」 | **必须改成「是/否」** |
| B4 | **是否做官方数据自监督** | 占位「是/否」 | **必须如实填** |
| B5 | 测试阶段后处理 | TTA + study 均值（患者平滑） | 与实际一致 |
| B6 | 是否使用测试集信息 | 否 | — |
| B7 | 训练数据范围 | 仅官方数据 | — |

---

## 3. 提交前检查清单

- [ ] `check_compliance.py` 全绿（体积/许可证/确定性）
- [ ] `cost.py` 跑出 **不一致 0 张**、平均耗时 **≤100ms**
- [ ] `submission.csv` 自检通过（14140 行、study 全覆盖、概率在 [0,1]）
- [ ] xlsx 的 **B3 / B4** 已按实际填写（不能留占位）
- [ ] `code/` 里的 `requirements.txt` 只列真实依赖
- [ ] 包内 `submission.csv` 是**历史最高分**的那一份
- [ ] 备份提交包与实验台账（日期/改动/valid 分/平台分）

---

## 4. 需要留意/待确认的风险点

1. **cost.py 的 study 覆盖问题**：100 张样本若未覆盖某个 study 的全部图片，study 级聚合无法
   精确重建。`cost.py` 会单独标注这类情况、不计入不一致。建议向主办方确认这 100 张是否
   覆盖完整 study；若不能，可考虑在提交说明里解释聚合口径。
2. **撤销机会只有一次**：换模型后分数可能短期下降，是否撤销当前 0.7936 的提交要谨慎，
   建议新方案在 valid 上稳定超过旧模型后再动。
3. **复赛容器依赖**：本目录只依赖 torch + torchvision，已规避 timm/albumentations 风险；
   但 `requirements.txt` 仍需与主办方镜像实际版本核对。
4. **预训练权重的 gated 风险**：torchvision 权重是公开直链、BSD 许可证，无 gated 问题；
   如果将来改用 HuggingFace 上的模型，务必确认不是 gated、不是 NC 许可证。
