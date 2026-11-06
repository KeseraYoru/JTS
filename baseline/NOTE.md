# 注意：这个目录现在只有数据，没有代码

`data600/` = 33984 张 600×600 预处理图（train 30344 / valid 1422 / test 2218），
是 `baseline1/` 与 `baseline3/` 训练时直接读的缓存。

## 为什么代码删了、目录名还留着

旧的非合规 pipeline（EVA-X + CheXFound）已整体删除，它的 `data600` 被两套新代码复用。
而 `baseline1/config.py` 与 `baseline3/config3.py` 的默认值都是：

```python
PREP_ROOT = os.path.join(_THIS, "..", "baseline", "data600")
```

**保留这个目录名，是为了不改两套代码的相对路径默认值。**
如果要把数据挪到别处，不要改代码，显式传环境变量：

```bash
CXR_PREP_ROOT=/新路径/data600 python train.py ...
```

## 重建方法（如果这份缓存丢了）

```bash
cd baseline1
python preprocess.py --size 600 --prep-root ../baseline/data600 --workers 16
```

需要 `../cxr_first/` 下的原图，33984 张，约 1~1.5 小时（CPU）。
