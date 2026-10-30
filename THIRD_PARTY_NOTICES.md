# 第三方代码与许可

本项目是面向 Fabric19 YOLO 数据的适配实现，不是官方 AnomalyNCD 的完整复制。

- **AnomalyNCD** — https://github.com/HUST-SLOW/AnomalyNCD ，参考提交 `9351c5daab246ff5db6841fde1689000d4f706f0`。MIT 许可见 `third_party/AnomalyNCD_LICENSE`。参考了数据组织、掩码引导表征学习、跨视图自蒸馏和区域面积合并流程。
- **DINO / DINOv1 ViT** — https://github.com/facebookresearch/dino 。`anomalyncd/mgvit.py` 基于 AnomalyNCD 的 `_MGViT.py` 修改，保留其 Facebook Apache-2.0 文件头；完整许可见 `third_party/DINO_LICENSE`。主要修改：移除硬编码 CUDA、保留 batch=1 维度、使用有限 attention mask 值、支持极小掩码和空掩码回退。
- **SimGCD** — https://github.com/CVMI-Lab/SimGCD 。多头原型分类器、对比学习与自蒸馏思路来自该 lineage；MIT 许可见 `third_party/SimGCD_LICENSE`。本仓库重新实现了 Fabric19 的已知/新类标签隔离和损失边界处理。

- **Ultralytics YOLO** — https://github.com/ultralytics/ultralytics 。端到端缺陷检测器使用其 YOLOv8 训练/推理接口；AGPL-3.0 许可见 `third_party/Ultralytics_LICENSE`。商业闭源使用请自行确认许可合规。

数据、DINO 预训练权重不随仓库分发。权重仍受其原始许可约束；本项目的代码许可不改变第三方权重的许可。
