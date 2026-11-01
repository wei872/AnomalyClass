# AnomalyClass — 基于 Ultralytics YOLO27 架构思想的布匹缺陷轻量化高精度检测模型

本仓库针对 **19 类工业布匹缺陷检测数据集**（[`dataset/fabric19.yaml`](dataset/fabric19.yaml)），融合 **[Ultralytics YOLO27 官方架构设计](https://docs.ultralytics.com/zh/models/yolo27)** 与 **YOLO26 预训练体系**，提供与 `YOLO26n` 参数量紧邻（融合后 **2.46M–2.51M** vs `YOLO26n` **2.38M**）且检测精度更优的三套模型方案：

1. **`models/yolo27n.yaml`（推荐默认高精度版）**：融合后 **2.512M 参数 / 6.63 GFLOPs**（仅比 `YOLO26n` 多 `0.13M` 参数），保留 `[P3/8, P4/16, P5/32]` 完整尺度覆盖并融入 YOLO27 小目标高分辨率加宽、可重参数化特征校准与前景对齐监督。
2. **`models/yolo27n-dual.yaml`（官方 YOLO27 双尺度版 `[P3/8, P5/32]`）**：融合后 **2.465M 参数 / 6.48 GFLOPs**（与 `YOLO26n` 参数量仅差 `3.6%`），移除中尺度检测头（Anchor 数 `8,400 -> 6,800`）并采用 `(1.0, 1/√2)` 双尺度平衡缩放。
3. **`models/yolo27n-lite.yaml`（工业边缘极轻量版）**：融合后仅 **0.776M 参数 / 3.29 GFLOPs**（参数量减少 **67%**），适合算力严苛的产线智能相机。

> **重要调用说明**：训练/验证/推理请务必使用本仓库提供的 `from yolo27 import YOLO27`（或 `python train_yolo27n.py`），**切勿直接用 `from ultralytics import YOLO("models/yolo27n.yaml")`**（原生 `YOLO` 类不会挂载 `YOLO27Detect`、前景对齐损失及 `yolo26n.pt` Net2Wider 预训练热启动）。

---

## 一、核心架构与训练机制设计（对齐官方 YOLO27 + 布匹缺陷特性）

### 1. DFL-Free 无约束边界回归（`reg_max: 1`）+ 100% 预训练回归头无损继承
- 与官方 `YOLO26 / YOLO27` 轻量 CNN 检测头保持一致，采用 `reg_max: 1` 直接回归归一化四边距离：
  - **消除大跨度细条瑕疵截断**：彻底解除 `reg_max=16` 在 `P3/8` 层 `15 × 8 = 120~240 px` 的回归跨度硬上限，使横跨整幅画面的细长瑕疵（*断纬、纬线不直、多股线、油点线、稀缝、包边不良*）可在高分辨率 `P3/8` 特征图上完整回归；
  - **Step-0 零误差继承 `yolo26n.pt` 检测头**：完整保留 `yolo26n.pt` 经 Objects365 + COCO 预训练的全部回归卷积（含 `.cv2.*.2`）与 `c3=80` 分类特征塔，避免通道截断或随机重置造成的定位退化。

### 2. 加宽早期高分辨率与小目标瓶颈阶段（Stronger Small-Object Backbone & Fine Neck）
- **`P2/4 (160×160)` 高分辨率阶段加宽**（`256 -> 320`，`n` 尺度 `64 -> 80 ch`）：在下采样前保留更细腻的经纬纱线纹理与边缘细节；
- **`P3/8 (80×80)` 小目标瓶颈容量翻倍**（`C3k2 [512, False, 0.50]`，内部瓶颈通道 `32 -> 64 ch`，对齐官方 `YOLO27-cls` 特征配比）：增强 *蚊虫小黑点、经线毛球、小尾巴、短废纱线* 等微小瑕疵的非线性表征能力，同时保持输出 `128 ch` 与预训练主干无缝对齐；
- **`P3/8 (80×80)` 细粒度颈部输出加宽**（`256 -> 320`，`80 ch`）：提升细粒度检测分支容量，并使分类头通道 `c3 = max(80, min(nc, 100)) = 80` 与 `yolo26n.pt` 1:1 精确匹配。

### 3. 固定 + 可学习逐通道多尺度特征校准（零推理开销重参数化折叠）
- 在 `YOLO27Detect`（[`yolo27/modules.py`](yolo27/modules.py)）中结合双尺度固定方差缩放系数（`[P3, P5]` 采用 `(1.0, 1/√2)`）与逐通道校准门控 `feat_logit_scale`（自动路由至无权重衰减、非 Muon 正交化参数组）；
- 调用 `fuse()` 推理或导出时，校准系数精确折叠进 `cv2`（`Conv2d`）与 `cv3`（`DWConv`）首层卷积核权重中，**部署阶段 0 额外参数、0 额外 FLOPs，融合前后误差 < 1e-4**。

### 4. 训练期活体颈部前景对齐监督 + 冲突消解一对一匹配（推理零开销剥离）
- **活体共享颈部前景对齐监督（Live Neck Foreground Alignment Supervision）**：`ForegroundAlignmentBranch` 直接作用于未截断梯度的共享颈部特征，以 `one2many` 的连续 IoU 质量为软目标计算 `Varifocal` 前景损失，强力拉大微小瑕疵与重复编织纹理背景的响应差距，并在 `fuse()` 时完全剥离（`fg_branch = None`）；
- **前景先验引导的一对一正样本分配 + 均衡双头进度（`o2m: 0.80 -> 0.55`）**：默认采用高精度 **一对多 + NMS**（`nms=True`）推理，同时以 `o2m + o2o = 1.0`（避免原生 `E2ELoss` 后期衰减至 `0.10` 导致一对多分支与主干欠拟合）及 `tal_topk=7, tal_topk2=1` + 前景先验引导训练 `one2one` 头，传入 `nms=False` 即可切换无 NMS 端到端推理。

---

## 二、模型复杂度对比（`nc=19`, 输入 `640×640`）

| 模型变体 | 配置文件 | 预测尺度 (Stride) | 训练态参数量 | 推理融合后参数量 | 融合后 FLOPs | 特点与适用场景 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **YOLO26n** (基准) | `yolo26n.yaml` | `[8, 16, 32]` | 2.511 M | 2.379 M | 5.33 B | 原生前代基准 |
| **YOLO27n** (推荐默认) | [`models/yolo27n.yaml`](models/yolo27n.yaml) | `[8, 16, 32]` | 2.736 M | **2.512 M** | **6.63 B** | 参数量贴近 YOLO26n（仅 `+0.13M`），小目标与细长条瑕疵精度最高 |
| **YOLO27n-Dual** (双尺度版) | [`models/yolo27n-dual.yaml`](models/yolo27n-dual.yaml) | `[8, 32]` | 2.630 M | **2.465 M** | **6.48 B** | 纯正官方 YOLO27 双尺度头，参数与 YOLO26n 几乎一致 |
| **YOLO27n-Lite** (极轻版) | [`models/yolo27n-lite.yaml`](models/yolo27n-lite.yaml) | `[8, 32]` | 0.860 M | **0.776 M** | **3.29 B** | 工业边缘相机极限高帧率部署 |

---

## 三、快速开始

### 1. 查看参数量与延迟对比
```bash
python train_yolo27n.py --mode profile
```

### 2. 训练模型（自动查找本地/同级目录 `yolo26n.pt` 执行无损 Net2Wider 热启动）
> 建议将你测试 `yolo26n` 时使用的 `yolo26n.pt` 放在项目根目录或上级目录（训练器会自动检索并打印 `YOLO27 Warm-Start: transferred ...` 日志），也可通过 `--donor path/to/yolo26n.pt` 显式指定：

```bash
# 1) 训练默认高精度 YOLO27n（推荐与 yolo26n 使用相同的 epochs/batch/imgsz 对比）
python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --epochs 150 --batch 16 --imgsz 640

# 2) 或训练双尺度版 YOLO27n-Dual
python train_yolo27n.py --mode train --model models/yolo27n-dual.yaml --data dataset/fabric19.yaml --epochs 150 --batch 16 --imgsz 640

# 3) Python 代码直接调用（注意：务必使用 YOLO27 而非 YOLO！）
python -c "from yolo27 import YOLO27; YOLO27('models/yolo27n.yaml').train(data='dataset/fabric19.yaml', epochs=150, imgsz=640, batch=16)"
```

### 3. 验证、推理与导出（支持 `nms=True` 默认高精度 / `nms=False` 端到端无 NMS）
```bash
# 默认：一对多检测头 + NMS（最高检测精度）
python train_yolo27n.py --mode val --weights runs/detect/yolo27n_fabric19/weights/best.pt --data dataset/fabric19.yaml --nms

# 可选：一对一端到端无 NMS 推理（nms=False）
python train_yolo27n.py --mode val --weights runs/detect/yolo27n_fabric19/weights/best.pt --data dataset/fabric19.yaml --no-nms

# 导出 ONNX（自动剥离前景对齐分支并折叠多尺度校准系数）
python train_yolo27n.py --mode export --weights runs/detect/yolo27n_fabric19/weights/best.pt --format onnx
python train_yolo27n.py --mode export --weights runs/detect/yolo27n_fabric19/weights/best.pt --format onnx --no-nms
```

### 4. 运行自动化验证单测
```bash
python tests/test_yolo27n.py
```
