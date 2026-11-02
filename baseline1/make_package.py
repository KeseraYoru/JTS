# -*- coding: utf-8 -*-
"""生成提交包 `队伍名_最终提交材料.zip`（规则附录 A 的结构）。

    python make_package.py --ckpt runs/xxx/ema_f12.pth --submission submission.csv \
        --team 我的队名

生成结构（与附录 A 示例一致）：
    队伍名_最终提交材料.zip
    ├── code/            全部源码 + requirements.txt
    ├── model/           最终推理权重
    ├── pretrain_model/  预训练权重说明（可选放实际文件）
    ├── licenses/        预训练模型许可证全文
    ├── cost.py          耗时与一致性核验脚本
    ├── 模型核验材料与数据申明清单.xlsx
    └── submission.csv

注意：只有「最终推理时实际加载的权重」计入 500MB 限制；预训练权重不参与推理，
不计入，但需随包提供或在清单里给出公网直链与哈希。
"""
import argparse
import hashlib
import os
import shutil
import zipfile

from config import PRETRAIN_SOURCES, TEAM_NAME

BSD3 = """BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

--- 说明 ---
本文件为 torchvision（含其提供的 ImageNet-1k 预训练权重）的 BSD-3-Clause
许可证全文。torchvision 源码与权重均以该许可证发布，允许包括商业用途在内的
再分发，不属于 NC（非商业）类许可证，也不是 gated 模型。
项目主页: https://github.com/pytorch/vision
"""


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def write_xlsx(path, ckpt, submission, pretrain_hash):
    """生成「模型核验材料与数据申明清单.xlsx」。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "模型核验与数据申明"

    header = ["序号", "名称", "规则依据", "内容/说明"]
    ws.append(header)
    for c in range(1, 5):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.alignment = Alignment(horizontal="center", vertical="center")

    size_mb = os.path.getsize(ckpt) / 1024 / 1024
    rows = [
        ("A1", "所用开源模型名称", "第 1 条", PRETRAIN_SOURCES[0]["name"]),
        ("A2", "权重公开链接", "第 1 条", PRETRAIN_SOURCES[0]["url"]),
        ("A3", "许可证全文", "第 1 条、附录 A",
         f"{PRETRAIN_SOURCES[0]['license']}，全文见 licenses/BSD-3-Clause_torchvision.txt"
         "（非 NC、非 gated，允许再分发）"),
        ("A4", "上游训练数据说明", "第 1 条",
         f"{PRETRAIN_SOURCES[0]['upstream_data']}；"
         f"model card: {PRETRAIN_SOURCES[0]['model_card']}"),
        ("A5", "预训练模型权重哈希", "第 1 条",
         pretrain_hash or "见公网直链，下载后计算 SHA-256"),
        ("A6", "推理精度", "第 7 条", "FP16（AMP 自动混合精度），提交声明同此"),
        ("B1", "最终推理权重文件", "第 8 条",
         f"{os.path.basename(ckpt)}，{size_mb:.1f}MB，SHA-256: {sha256_file(ckpt)}"),
        ("B2", "推理时加载的权重合计", "第 8 条",
         f"{size_mb:.1f}MB ≤ 500MB"),
        ("B3", "验证集是否并入训练", "第 3 条",
         "见训练记录；并入前已完成全部超参与 ckpt 选择"),
        ("B4", "是否在官方训练集上自监督预训练", "补充四",
         "是/否（若使用：仅官方 train split，方式 SimCLR，无外部数据）"),
        ("B5", "测试阶段后处理", "第 6 条",
         "TTA（6 种确定性变换取平均）+ 同一 study 多图预测取均值（患者平滑），"
         "均为确定性聚合、不拟合参数"),
        ("B6", "是否使用测试集信息", "第 5 条",
         "否：未使用测试集做任何统计量估计、阈值拟合或模型选择"),
        ("B7", "训练数据范围", "第 2 条",
         "仅使用赛事官方提供数据；未使用任何外部医学数据（含无标签）"),
    ]
    for r in rows:
        ws.append(list(r))

    widths = [6, 30, 16, 80]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    # 第二页：推理耗时说明
    ws2 = wb.create_sheet("推理耗时与一致性")
    ws2.append(["项目", "说明"])
    for c in range(1, 3):
        ws2.cell(row=1, column=c).font = Font(bold=True)
    ws2.append(["计时口径", "100 张、batch=1 串行、连续 3 轮；每张取 3 轮均值后对 100 张平均"])
    ws2.append(["上限", "单张端到端 ≤100ms（含读图与预处理，不含模型加载/预热）"])
    ws2.append(["实测结果", "见 cost.py 输出的 [CONCLUSION] 行"])
    ws2.append(["一致性要求", "与 submission.csv 样本级一致；|Δp|≤1e-3 视为浮点波动，不一致张数须为 0"])
    ws2.column_dimensions["A"].width = 18
    ws2.column_dimensions["B"].width = 90

    wb.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="最终推理权重")
    parser.add_argument("--submission", required=True, help="最终 submission.csv")
    parser.add_argument("--team", default=TEAM_NAME)
    parser.add_argument("--pretrain", default=None,
                        help="可选：预训练权重文件（放 pretrain_model/，不计入 500MB）")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    pkg_name = f"{args.team}_最终提交材料"
    out_dir = args.out_dir or os.path.join(here, "package")
    root = os.path.join(out_dir, pkg_name)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)

    # ---------- code/ ----------
    code_dir = os.path.join(root, "code")
    os.makedirs(code_dir)
    n_py = 0
    for fn in sorted(os.listdir(here)):
        if fn.endswith(".py"):
            shutil.copy2(os.path.join(here, fn), code_dir)
            n_py += 1
    for fn in ("requirements.txt",):
        p = os.path.join(here, fn)
        if os.path.exists(p):
            shutil.copy2(p, code_dir)

    # ---------- model/ ----------
    model_dir = os.path.join(root, "model")
    os.makedirs(model_dir)
    shutil.copy2(args.ckpt, model_dir)

    # ---------- pretrain_model/ ----------
    pre_dir = os.path.join(root, "pretrain_model")
    os.makedirs(pre_dir)
    pretrain_hash = None
    if args.pretrain and os.path.exists(args.pretrain):
        shutil.copy2(args.pretrain, pre_dir)
        pretrain_hash = sha256_file(args.pretrain)
    with open(os.path.join(pre_dir, "说明.md"), "w", encoding="utf-8") as f:
        f.write(
            "# 预训练权重说明\n\n"
            f"- 名称：{PRETRAIN_SOURCES[0]['name']}\n"
            f"- 公网直链：{PRETRAIN_SOURCES[0]['url']}\n"
            f"- 许可证：{PRETRAIN_SOURCES[0]['license']}\n"
            f"- 上游数据：{PRETRAIN_SOURCES[0]['upstream_data']}\n"
            f"- model card：{PRETRAIN_SOURCES[0]['model_card']}\n"
            f"- SHA-256：{pretrain_hash or '（未随包附带，见直链）'}\n\n"
            "该权重不参与最终推理（推理只加载 model/ 下的微调权重），不计入 500MB 限制。\n"
        )

    # ---------- licenses/ ----------
    lic_dir = os.path.join(root, "licenses")
    os.makedirs(lic_dir)
    with open(os.path.join(lic_dir, "BSD-3-Clause_torchvision.txt"), "w",
              encoding="utf-8") as f:
        f.write(BSD3)

    # ---------- cost.py（规则要求放在包根目录）----------
    shutil.copy2(os.path.join(here, "cost.py"), root)
    # ---------- submission.csv ----------
    shutil.copy2(args.submission, root)
    # ---------- 清单 xlsx ----------
    write_xlsx(os.path.join(root, "模型核验材料与数据申明清单.xlsx"),
               os.path.join(model_dir, os.path.basename(args.ckpt)),
               args.submission, pretrain_hash)

    # ---------- 打包 ----------
    zip_path = os.path.join(out_dir, f"{pkg_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, out_dir)
                zf.write(full, rel)

    print(f"✓ 代码 {n_py} 个 .py 已打包")
    print(f"✓ 提交包已生成: {zip_path}")
    print(f"  大小 {os.path.getsize(zip_path) / 1024 / 1024:.1f} MB")
    print("包含：code/ model/ pretrain_model/ licenses/ cost.py "
          "模型核验材料与数据申明清单.xlsx submission.csv")


if __name__ == "__main__":
    main()
