#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
screen_solution.py —— Kaggle 方案「能不能移植到本地赛」体检脚本

用途：拿到任何一份公开方案（notebook / 仓库）后，先跑这个脚本，
      按《比赛参赛规则》第 1/2/5/6/7/8 条与《规则补充》二~九，
      一次性把「不可修复的红线」和「必须改的适配点」列出来。

用法：
    python3 screen_solution.py <目录 | .ipynb | .py>
    python3 screen_solution.py ../Grand-X-Ray-Slam-competition

只读不写，不改动任何原文件。纯标准库，无需安装依赖。

输出三档：
    [红线] 整份方案作废或必须换骨架，改代码没用
    [适配] 可以改，但漏改会静默掉分或提交失败
    [提示] 需要人工确认
"""

import json
import os
import re
import sys

CJK = re.compile(r"[\u4e00-\u9fff]")

# ---------------------------------------------------------------- 规则库
# (编号, 严重度, 名称, 正则, 判定理由, 处置建议)
RULES = [
    # ========== 一、预训练权重：规则第 1 条 + 补充二 ==========
    ("W1", "红线", "医学影像预训练权重",
     r"(eva[-_ ]?x|evax|chexfound|chexpert|mimic|rad[-_ ]?imagenet|radimagenet|"
     r"rad[-_ ]?dino|raddino|torchxrayvision|\bxrv\b|chexnet|medclip|biomedclip|"
     r"cxr[-_ ]?foundation|cxrfoundation|retfound|medsam|gloria|ark\+|"
     r"medvit|medicalnet|modelcard|biovil|biomed\w*clip)",
     "预训练主数据集若是医学影像/文献，整份方案违反规则第 1 条，且不可通过改代码修复",
     "整份方案降级为「只借思路」，骨干换成 torchvision ImageNet 权重"),

    ("W2", "红线", "从网上现场下载权重",
     r"(^\s*!\s*(pip|wget|curl|gdown|apt)|hub\.load|torch\.hub\.load_state_dict|"
     r"gdown\.download|requests\.get\([^)]*\.(pth|pt|bin|safetensors|ckpt))",
     "复赛容器离线且只保证 torch+torchvision；运行期下载=必然失败，且外部权重来源不可申报",
     "权重必须提前落盘进 pretrain_model/，并写进申报表 A1~A6"),

    ("W3", "提示", "transformers / open_clip 拉取模型",
     r"(from_pretrained\s*\(|open_clip\.create_model|AutoModel\.from_pretrained|"
     r"timm\.create_model)",
     "写法本身合规与否取决于模型名，必须逐个核对权重卡（model card）",
     "确认预训练主数据集是 ImageNet/LAION 等通用数据；且注意镜像里没有 transformers"),

    # ========== 二、外部数据：规则第 2 条 + 补充五 ==========
    ("D1", "红线", "引用外部医学数据集",
     r"(nih[_-]?chest|chestx-?ray14|chestxray14|padchest|vindr|mimic[-_ ]?cxr|"
     r"openi[_-]?dataset|rsna[_-]?pneumonia|tbx11k|jsrt|chestx-det)",
     "规则第 2 条规定不得用任何非赛方医学数据（含无标签数据）做预训练/续训/伪标签/微调",
     "删除该数据源；若它只用于「算 mean/std」，同样违规（属外部数据统计量）"),

    ("D2", "红线", "硬编码外部数据统计量",
     r"\b(mean|std|MEAN|STD)\b\s*=\s*[\(\[]\s*-?[0-9.]",
     "灰度 X 光常用单通道均值方差；若该数值是在 CheXpert/MIMIC 上统计得来即违规，"
     "ImageNet 自带常量可用",
     "改成在官方训练集上统计（合规）或直接用 0.5/0.5，并把选择理由写进文档",
     # 排除 ImageNet 标准三通道常量（合规）
     r"0\.485\s*,\s*0\.456\s*,\s*0\.406|0\.229\s*,\s*0\.224\s*,\s*0\.225"),

    # ========== 三、测试集使用：规则第 5 条 ==========
    ("T1", "红线", "在 test 上选模型/拟合统计量",
     r"(test_df\[[^\]]*\]\.(mean|std|describe)|pseudo[_-]?label|"
     r"(best|select|choose|pick)[_a-z]{0,20}.{0,30}(test_?(acc|auc|score|f1|loss)))",
     "规则第 5 条明确禁止测试集参与 checkpoint 选择、阈值、归一化统计量、伪标签",
     "删除；所有选择改在自建 patient 级验证集上做"),

    # ========== 四、依赖：容器只有 torch+torchvision ==========
    ("E1", "适配", "非 torch/torchvision 依赖",
     r"(^\s*(import|from)\s+(timm|albumentations|cv2|fvcore|open_clip|"
     r"transformers|einops|sklearn|scipy|pandas|segmentation_models_pytorch|"
     r"apex|xformers|deepspeed|lightgbm|tqdm))",
     "复赛容器只保证 PyTorch 2.10 + torchvision 0.25（离线）；numpy/PIL 随 torch 自带，"
     "但 timm/albumentations/cv2/fvcore/open_clip/transformers 大概率没有",
     "能删就删：timm→torchvision，albumentations→transforms.v2，cv2→PIL，"
     "scipy/sklearn→numpy 手写（AUC 已自写）"),

    # ========== 五、标签体系：14 类 → 10 类 ==========
    ("L1", "适配", "14 类标签名硬编码",
     r"(Pleural Other|Lung Lesion|Support Devices|Fracture|Enlarged Cardiomediastinum)",
     "Kaggle 版是 MIMIC 14 类；本地只有 10 类，且本地类别 id 是数字 1~10（顺序≠字母序）",
     "按名映射到 id，删掉 Fracture/Lung Lesion/Pleural Other/Support Devices 四列，"
     "严禁按列位置直接切片（会整体错位）"),

    ("L2", "适配", "类别数量/顺序定义",
     r"\b(num_classes|NUM_CLASSES|TARGET_CLASSES|n_classes|labels_cols|LABELS|CLASSES)\b\s*=",
     "标签顺序直接决定「第 k 个输出对应哪个病」；Kaggle 用字母序 14 类，本地是数字 id 1~10",
     "写一张显式映射表并用单元测试校验：10 个 id × 正样本数 对得上"),

    ("L3", "提示", "阈值 / softmax 逻辑",
     r"(threshold\s*=\s*0?\.[0-9]|>=\s*0\.[0-9]|softmax|argmax|topk)",
     "Macro AUC 与阈值无关；若方案按阈值截断输出二值、或用 softmax 强制单标签，会毁掉排序",
     "只输出概率，不做二值截断；类别 10 与其余 9 类互斥可作为结构先验保留"),

    # ========== 六、提交格式与粒度 ==========
    ("S1", "适配", "Kaggle 宽表提交",
     r"(Image_name|sample_submission|dicom_id|to_csv\()",
     "Kaggle 是「一行一图 + 14 列概率」的宽表，且评的是图像级；"
     "本地是「一行一类」的长表，评的是 study 级",
     "重写提交层：图级概率 → 按 Study_id 聚合（均值）→ 展开成长表 → 保留 6 位小数"),

    ("S2", "提示", "study 级聚合缺失",
     r"(groupby|\.mean\(\)|pivot_table)",
     "本地评分单位是 Study_id，一次检查可能多张图；Kaggle 按图评，方案里通常没有聚合步骤",
     "补上 study 聚合，并用自建验证集确认 mean/max 哪个更好（该选择须在并入验证集前定死）"),

    # ========== 七、路径 ==========
    ("P1", "适配", "Kaggle 路径",
     r"(/kaggle/input|/kaggle/working|/content/|\.\./input/)",
     "路径全部失效；本地是 cxr_first/{train,valid,test}/subject_x/study_y/pic_z.jpg 三层结构",
     "改成从 CSV 的 Subject_id/Study_id 拼路径，且 Study_id 按字符串读（保留 7 位前导零）"),

    # ========== 八、复赛硬约束 ==========
    ("Z1", "适配", "大模型 / 大分辨率",
     r"(vit[-_]?l(arge)?|convnext[-_]?large|swin[-_]?l|eva[-_]?l|beit[-_]?l|"
     r"resnet(101|152)|densenet(161|169|201)|512|600|768)",
     "权重解压后 ≤500MB、单张端到端 ≤100ms（T4）。ViT-L 类 fp32 就 1.2GB，直接超",
     "选 ≤100M 参数的骨干（convnext_base 87.6M/334MB 是目前验证过的上限附近），"
     "分辨率 384 起步，用 cost.py 实测再定"),

    ("Z2", "提示", "TTA / 集成次数",
     r"(hflip|flip|tta|TTA|ensemble|fold)",
     "TTA 与集成本身合规（规则第 6 条），但每次 TTA 都乘时延；集成还要乘体积",
     "先算预算：TTA 次数 × 单次时延 ≤ 100ms，集成模型总体积 ≤ 500MB"),
]


def read_code(path):
    """返回 [(行号, 代码行)]；notebook 只取 code cell，跳过输出。"""
    if path.endswith(".ipynb"):
        try:
            nb = json.load(open(path, encoding="utf-8"))
        except Exception:
            return []
        lines = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for item in cell.get("source", []):
                lines.extend(item.splitlines())
        return list(enumerate(lines, 1))
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return []
    return list(enumerate(text.splitlines(), 1))


def collect(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
        for f in files:
            if f.endswith((".py", ".ipynb")):
                out.append(os.path.join(root, f))
    return sorted(out)


def screen(files, quiet=False):
    hits = {}   # rule_id -> [(file, line, text)]
    compiled = []
    for r in RULES:
        rid, sev, name, pat, why, fix = r[:6]
        skip = re.compile(r[6], re.I) if len(r) > 6 else None
        compiled.append((rid, sev, name, re.compile(pat, re.I), why, fix, skip))

    for f in files:
        for no, line in read_code(f):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # 去掉行尾注释，再判断是不是「纯说明文字」（没有 = ( import 的一律当散文跳过），
            # 避免中文注释/文档字符串里提到 EVA-X 就被误判成使用了医学权重
            code = s.split("#")[0].strip()
            if not code:
                continue
            # 中文占比高、或以中文标点结尾的行是文档字符串/说明文字（哪怕带 = 和括号），跳过
            if code.endswith(("。", "；", "，", "：", "、", "）")):
                continue
            if len(CJK.findall(code)) / max(len(code), 1) > 0.12:
                continue
            if not re.search(r"[=(\[]|^\s*(import|from)\b", code):
                continue
            for rid, sev, name, rx, why, fix, skip in compiled:
                if rx.search(code) and not (skip and skip.search(code)):
                    hits.setdefault(rid, []).append((f, no, s))

    order = {r[0]: i for i, r in enumerate(RULES)}
    meta = {r[0]: r for r in RULES}

    print("=" * 74)
    print(f"体检对象：{len(files)} 个代码文件")
    print("=" * 74)
    for rid in sorted(hits, key=lambda r: order[r]):
        sev, name, why, fix = meta[rid][1], meta[rid][2], meta[rid][4], meta[rid][5]
        locs = hits[rid]
        print(f"\n[{sev}] {rid} {name}   —— 命中 {len(locs)} 处 / "
              f"{len({l[0] for l in locs})} 个文件")
        print(f"   为什么：{why}")
        print(f"   怎么办：{fix}")
        if not quiet:
            for f, no, s in locs[:2]:
                print(f"     · {os.path.basename(f)}:{no}  {s[:88]}")
            if len(locs) > 2:
                print(f"     · ... 其余 {len(locs) - 2} 处略")

    reds = [rid for rid in hits if meta[rid][1] == "红线"]
    reds.sort(key=lambda r: order[r])
    print("\n" + "=" * 74)
    print("汇总：" + ("  ".join(f"{rid}{'🔴' if meta[rid][1] == '红线' else '🟡'}"
                              for rid in sorted(hits, key=lambda r: order[r]))
                     or "无命中"))
    print("-" * 74)
    if reds:
        print(f"结论：{len(reds)} 项红线（{', '.join(reds)}）→ 不能直接移植，最多借思路")
        print("      红线均属「换个骨干 = 重做方案」，别再花时间改这份代码的模型部分")
    elif hits:
        print("结论：无红线 → 可移植，但须逐项完成适配并跑移植自检清单")
    else:
        print("结论：未命中规则 → 仍要人工核对权重来源与提交格式两头")
    print("=" * 74)
    return 1 if reds else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    target = sys.argv[1]
    quiet = "--quiet" in sys.argv
    files = collect(target)
    if not files:
        print(f"没有找到 .py/.ipynb：{target}")
        sys.exit(2)
    sys.exit(screen(files, quiet))
