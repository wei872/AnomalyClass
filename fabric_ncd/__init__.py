"""FabricNCD: 轻量化布匹疵点检测 / 分类 / 新类发现.

参考 AnomalyNCD (CVPR 2025) 的多类异常分类框架, 针对机织布匹 (经纬纹理、
布边/布中纹理不一致、布幅可变) 做了定向改造:

* 全局 + 局部双分支特征 (``models/glocal.py``)
* 经纬纹理先验 (Gabor 滤波器组 + 织造周期估计, ``models/texture.py``)
* 布面前景 / 背景分离与布边感知 (``models/heads.py`` + 布幅增强 ``data/augment.py``)
* 正常样本记忆库 + 学习式异常图 (``models/memory.py``)
* AnomalyNCD 式主元素二值化 / 掩码引导表征 / 区域合并 (``ncd.py``)

主入口见 ``tools/train.py`` / ``tools/evaluate.py`` / ``tools/infer.py``.
"""

from .version import __version__  # noqa: F401

__all__ = ["__version__"]
