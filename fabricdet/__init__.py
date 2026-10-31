# FabricDefectNet: 轻量型端到端布匹缺陷 检测-定位-分类-未知缺陷识别 一体化模型
# 骨干: DINOv2 ViT-S/14 (官方最轻量档), 颈: 轻量上采样 FPN, 头: 锚_free 密集头 + 原型度量分类头(开集)
__version__ = "1.0.0"

from .model import FabricDefectNet  # noqa: F401
