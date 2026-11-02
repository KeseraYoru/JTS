#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CheXFound 环境检查脚本。

一键验证训练 CheXFound 需要的一切是否就位：
  1. 第三方代码目录（third_party/CheXFound，git clone 得到）
  2. 4 个官方权重/配置文件（本地下载后人工放入 third_party/CheXFound_Model/）
  3. Python 依赖（fvcore / open_clip / omegaconf / skmultilearn / torchmetrics /
     iopath / pandas / transformers / scikit-image——对 chexfound 源码做静态
     import 分析得出，覆盖训练完整导入链）
  4. chexfound 包及训练用到的子模块能否导入

用法（在 baseline/ 目录下）：
    python3 check_chexfound.py

全部通过会提示可以开训；任何一项失败都会给出具体的补救命令。
退出码：0 = 全部通过，1 = 有未解决项。

注意：
- 包的「发行名」和「导入名」可能不同：pip 装的是 open_clip_torch，
  import 的是 open_clip（本脚本已处理）；
- 官方 conda-extras.yaml 里的 xformers 不用装：代码会自动回退到 PyTorch 原生
  attention（只有 warning），而 xformers 0.0.18 与 torch 2.14/CUDA 13 不兼容，
  强行装只会把环境搞坏。
"""
import os
import sys

# 路径相对本脚本所在目录解析，从任何工作目录跑都准
HERE = os.path.dirname(os.path.abspath(__file__))
CHEXFOUND_REPO = os.environ.get("CXR_CHEXFOUND_REPO",
                                os.path.join(HERE, "third_party", "CheXFound"))
CHEXFOUND_MODEL_DIR = os.environ.get(
    "CXR_CHEXFOUND_MODEL_DIR",
    os.path.join(HERE, "third_party", "CheXFound_Model"))

# (文件名, 最小合理字节数, 最大合理字节数, 说明)
REQUIRED_FILES = [
    ("config.yaml", 100, 50 * 1024 * 1024, "模型结构配置"),
    ("teacher_checkpoint.pth", 50 * 1024 * 1024, 5 * 1024 ** 3,
     "自监督预训练 backbone 权重（ViT-Large，最大的一件）"),
    ("glori.pth", 100 * 1024, 2 * 1024 ** 3, "GLoRIA 线性分类头权重"),
    ("results_eval_linear.json", 100, 50 * 1024 * 1024,
     "最佳分类器配置（分类头的结构参数从这里读）"),
]

# (导入名, pip 包名) —— 两组「发行名≠导入名」的坑：
#   pip 装 open_clip_torch，import 的是 open_clip
#   pip 装 scikit-multilearn，import 的是 skmultilearn
# 这份清单是对 chexfound 源码做静态 import 分析得出的（2026-10-21），
# 覆盖 train.py --model chexfound 的完整导入链；peft/cuml/submitit/toolz
# 只被官方完整训练脚本（linear_glori.py / run/*）使用，我们不跑那些，不需要装。
PYTHON_DEPS = [
    ("fvcore", "fvcore"),
    ("open_clip", "open_clip_torch"),
    ("omegaconf", "omegaconf"),
    ("skmultilearn", "scikit-multilearn"),
    ("torchmetrics", "torchmetrics"),
    ("iopath", "iopath"),
    ("pandas", "pandas"),
    ("transformers", "transformers"),
    ("skimage", "scikit-image"),
]

# (模块, 训练是否必需) —— chexfound.data.transforms 只有 Kaggle notebook 的
# 推理代码用，我们的训练不导入，缺了只警告不拦路
CHEXFOUND_MODULES = [
    ("chexfound", True),
    ("chexfound.eval.setup", True),                  # setup_and_build_model
    ("chexfound.eval.utils", True),                  # extract_hyperparameters_from_model
    ("chexfound.eval.classification.utils", True),   # setup_glori
    ("fvcore.common.checkpoint", True),              # Checkpointer（加载 glori.pth）
    ("chexfound.data.transforms", False),            # 可选：需要 pandas + scikit-image
]

PASS, FAIL = "\033[32m✓\033[0m", "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"


def ok(msg):
    print(f"  {PASS} {msg}")


def bad(msg):
    print(f"  {FAIL} {msg}")


def warn(msg):
    print(f"  {WARN} {msg}")


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n:.0f}B"
        n /= 1024


def check_repo():
    """第 1 项：第三方代码目录。"""
    print(f"\n[1/4] 第三方代码目录: {CHEXFOUND_REPO}")
    if not os.path.isdir(CHEXFOUND_REPO):
        bad("目录不存在")
        print("      补救：git clone --recursive https://github.com/RPIDIAL/CheXFound.git "
              f"{CHEXFOUND_REPO}")
        print("      或手动解压你下载的 CheXFound 压缩包到该路径")
        return False
    pkg = os.path.join(CHEXFOUND_REPO, "chexfound")
    if not os.path.isdir(pkg):
        bad("目录存在，但没有 chexfound/ 包（可能下错了或只下了一部分）")
        print(f"      补救：确认 {CHEXFOUND_REPO} 里直接能看到 chexfound/ 目录")
        return False
    if not os.path.isdir(os.path.join(CHEXFOUND_REPO, ".git")):
        warn("不是 git clone 下来的（没有 .git）——只要 chexfound/ 包完整就能用，")
        print("      但建议用 git clone 保证拿全文件")
    ok(f"chexfound/ 包就位（{human(_dir_size(pkg))}）")
    return True


def check_files():
    """第 2 项：4 个权重/配置文件。"""
    print(f"\n[2/4] 权重与配置文件: {CHEXFOUND_MODEL_DIR}")
    if not os.path.isdir(CHEXFOUND_MODEL_DIR):
        bad("目录不存在")
        print("      补救：mkdir -p " + CHEXFOUND_MODEL_DIR)
        print("      然后把本地下载好的 4 个文件放进去（见下方清单）")
        return False
    all_ok = True
    for fname, min_b, max_b, desc in REQUIRED_FILES:
        path = os.path.join(CHEXFOUND_MODEL_DIR, fname)
        if not os.path.exists(path):
            bad(f"{fname} 缺失（{desc}）")
            all_ok = False
            continue
        size = os.path.getsize(path)
        if size < min_b:
            bad(f"{fname} 只有 {human(size)}，太小了（{desc}，至少 {human(min_b)}）")
            if fname.endswith(".pth"):
                print("      .pth 太小通常是下载失败时把网页错误页存成了权重文件，")
                print("      请重新下载（浏览器登录 Google Drive 后下载，不要用脚本）")
            all_ok = False
            continue
        if size > max_b:
            warn(f"{fname} 有 {human(size)}，比预期大不少，确认没下错文件")
        # torch.save 的 .pth 是 zip 格式，头两个字节是 PK；
        # 不是 PK 基本可以断定是 HTML/txt 错误页
        if fname.endswith(".pth") and not _is_zip(path):
            bad(f"{fname} 不是有效的 torch 权重文件（疑似网页错误页/文本文件）")
            print("      用文本编辑器打开看一眼，如果是 HTML 就是下载失败，重新下载")
            all_ok = False
            continue
        ok(f"{fname}（{human(size)}）")
    if not all_ok:
        print("\n      这 4 个文件要从 Google Drive 官方文件夹下载：")
        print("      https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806")
        print("      （国内网络打不开就找台能访问的设备下载，再传到服务器该目录）")
    return all_ok


def check_python_deps():
    """第 3 项：chexfound 导入链需要的 Python 包。"""
    print("\n[3/4] Python 依赖")
    missing = []
    for mod, pip_name in PYTHON_DEPS:
        try:
            __import__(mod)
            ok(f"{pip_name}（import {mod}）")
        except ImportError as e:
            bad(f"{pip_name} 导入失败: {e}")
            missing.append(pip_name)
    if missing:
        print(f"\n      一次性补齐：pip install {' '.join(missing)}"
              " -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("      （xformers 不用装：代码自动回退原生 attention，且官方 pin 的"
              "0.0.18 与 torch 2.14 不兼容）")
    return not missing


def check_chexfound_importable():
    """第 4 项：chexfound 包及训练用到的子模块。"""
    print("\n[4/4] chexfound 包及子模块")
    if CHEXFOUND_REPO not in sys.path:
        sys.path.insert(0, CHEXFOUND_REPO)
    all_ok = True
    for mod, required in CHEXFOUND_MODULES:
        try:
            __import__(mod)
            ok(f"{mod}" + ("" if required else "（可选）"))
        except ImportError as e:
            msg = f"{mod} 导入失败: {e}"
            if required:
                bad(msg)
                print(f"      缺依赖就按 [3/4] 的组合命令装；若 {CHEXFOUND_REPO} 下代码")
                print("      不完整，重新 git clone --recursive")
                all_ok = False
            else:
                warn(msg)
                print("      我们训练不导入它，可暂不处理（要消除警告就装 pandas + scikit-image）")
    return all_ok


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _is_zip(path):
    """torch.save 默认 zip 格式，前 4 字节是 PK\\x03\\x04。"""
    try:
        with open(path, "rb") as f:
            return f.read(4)[:2] == b"PK"
    except OSError:
        return False


def main():
    print("=" * 62)
    print("CheXFound 环境检查")
    print(f"代码目录: {CHEXFOUND_REPO}")
    print(f"权重目录: {CHEXFOUND_MODEL_DIR}")
    print("=" * 62)

    results = [
        check_repo(),
        check_files(),
        check_python_deps(),
        check_chexfound_importable(),
    ]
    passed = sum(1 for r in results if r)
    print("\n" + "=" * 62)
    if passed == len(results):
        print(f"全部 {passed}/{len(results)} 项通过 —— 可以运行：")
        print("    python3 train.py --model chexfound --epochs 6")
        sys.exit(0)
    print(f"通过 {passed}/{len(results)} 项，按上面的补救命令逐项解决后重跑本脚本。")
    sys.exit(1)


if __name__ == "__main__":
    main()
