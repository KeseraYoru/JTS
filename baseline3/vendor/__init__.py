# -*- coding: utf-8 -*-
"""vendor 包：把外部架构的**前向实现**内置进来。

为什么需要 vendor
----------------
复赛容器只保证 torch + torchvision。timm 不在里面，而我们要用的
ImageNet-22k ConvNeXt 与 DINOv2 ViT 都需要 timm 才能构建。
所以把这两份前向代码（纯 torch，无第三方依赖）复制进 code/，
权重以 .pth 形式随包提交 —— 容器里只 import 本目录即可。

合规性
------
这两份实现只是「网络结构的代码」，不含任何预训练知识；
预训练知识全部来自 weights/*.pth，其来源与许可证在 config3.PRETRAIN_SOURCES 中登记。
"""
