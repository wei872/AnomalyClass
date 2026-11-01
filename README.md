# AnomalyClass — 基于 Ultralytics YOLO27 架构思想的布匹缺陷轻量化高精度检测模型

本仓库针对 **19 类工业布匹缺陷检测数据集**（[`dataset/fabric19.yaml`](dataset/fabric19.yaml)），融合 **[Ultralytics YOLO27 官方架构设计](https://docs.ultralytics.com/zh/models/yolo27)** 与 **YOLO26 预训练热启动体系**，提供与 `YOLO26n` 参数量紧邻（融合后 **2.45M–2.49M** vs `YOLO26n` **2.38M**）且泛化与检测精度更优的三套模型方案：

1. **`models/yolo27n.yaml`（推荐默认高精度版）**：融合后 **2.491M 参数 / 6.55 GFLOPs**（仅比 `YOLO26n` 多 `0.11M` 参数），保留 `[P3/8, P4/16, P5/32]` 完整尺度覆盖，融合高分辨率小目标加宽、固定跨尺度折叠校准与**背景改变—疵点不变前景对齐辅助监督**。
2. **`models/yolo27n-dual.yaml`（官方 YOLO27 双尺度版 `[P3/8, P5/32]`）**：融合后 **2.450M 参数 / 6.42 GFLOPs**（与 `YOLO26n` 参数量仅差 `3.0%`），移除中尺度检测头（Anchor 数 `8,400 -> 6,800`）并采用 `(1.0, 1/√2)` 双尺度平衡缩放 + 前景对齐辅助监督。
3. **`models/yolo27n-lite.yaml`（工业边缘极轻量版）**：融合后仅 **0.776M 参数 / 3.29 GFLOPs**（参数量减少 **67%**），同样配备训练期前景对齐辅助监督并在推理时零开销剥离，适合算力严苛的产线智能相机。

> **重要调用说明**：训练/验证/推理请务必使用本仓库提供的 `from yolo27 import YOLO27`（或 `python train_yolo27n.py`），**切勿直接用 `from ultralytics import YOLO("models/yolo27n.yaml")`**（原生 `YOLO` 类不会挂载 `YOLO27Detect`、前景对齐辅助监督及 `yolo26n.pt` Net2Wider 预训练热启动）。

---

## 一、核心架构与训练机制设计（对齐官方 YOLO27 + 布匹缺陷特性）

### 1. DFL-Free 无约束边界回归（`reg_max: 1`）+ 1:1 预训练检测头无损继承
- 与官方 `YOLO26 / YOLO27` 轻量 CNN 检测头保持一致，采用 `reg_max: 1`、`c2=16`、`c3=80` 直接回归归一化四边距离 + 各向异性细条边缘精准损失（`YOLO27BboxLoss`）：
  - **消除大跨度细条瑕疵截断**：解除 `reg_max=16` 在 `P3/8` 层 `15 × 8 = 120 px` 的回归跨度硬上限，使横跨整幅画面的细长瑕疵（*断纬、纬线不直、多股线、油点线、稀缝、包边不良*）可在高分辨率 `P3/8` 特征图上完整且紧贴边缘回归；
  - **Step-0 零误差继承 `yolo26n.pt` 检测头（覆盖率 `91.6%`）**：保持 `c2=16`、`c3=80` 与 `yolo26n.pt` 1:1 对齐，避免在小样本类别（12~20 框/类）上向回归塔注入随机未预训练卷积核。

### 2. 加宽早期高分辨率与小目标瓶颈阶段（Stronger Small-Object Backbone & Fine Neck）
- **`P2/4 (160×160)` 高分辨率阶段加宽**（`256 -> 320`，`n` 尺度 `64 -> 80 ch`）：在下采样前保留更细腻的经纬纱线纹理与边缘细节；
- **`P3/8 (80×80)` 小目标瓶颈容量翻倍**（`C3k2 [512, False, 0.50]`，内部瓶颈通道 `32 -> 64 ch`）：增强 *蚊虫小黑点、经线毛球、小尾巴、短废纱线、飘絮* 等微小瑕疵的非线性表征能力，同时通过零冲击 Net2Wider 切片保持 Step-0 等价；
- **`P3/8 (80×80)` 细粒度颈部输出加宽**（`256 -> 320`，`80 ch`）：提升细粒度检测分支容量，并使分类头通道 `c3 = 80` 与 `yolo26n.pt` 精确匹配。

### 3. 确定性固定跨尺度特征平衡缩放（零 EMA/BN 漂移 + 零推理开销折叠）
- 在 `YOLO27Detect`（[`yolo27/modules.py`](yolo27/modules.py)）中采用无状态确定性固定跨尺度缩放因子（双尺度 `[P3, P5]` 采用 `(1.0, 1/√2)`，三尺度采用 `(1.0, 1.0, 1.0)`），确保 `ModelEMA` 权重与 `BatchNorm2d` 统计量 100% 同步；
- 调用 `fuse()` 推理或导出时，缩放系数代数折叠进 `cv2`（`Conv2d`）与 `cv3`（`DWConv`）首层卷积核权重中，**部署阶段 0 额外参数、0 额外 FLOPs，融合前后误差 < 1e-4**。

### 4. 前景对齐辅助监督（背景发生改变，疵点不变，泛化能力增强 —— 训练期启用，推理剥离）
在 `YOLO27n`、`YOLO27n-Dual`、`YOLO27n-Lite` 三套模型中全面启用**端到端配对式“背景发生改变、疵点不变”前景对齐辅助监督**（`YOLO26n` 无该机制）：
- **配对图像构造：无空洞底纹修补交换 + 真值疵点逐比特锁死（[`yolo27/augment.py`](yolo27/augment.py)）**：
  - 训练期在 `YOLO27DetectionModel.loss(batch)` 内根据当前批次原图 $I$ 与真值框 $G$ 实时构造配对背景扰动图 $\tilde{I}$：先用同幅布面的平移洁净底纹修补供体图中的缺陷区域（避免跨样本交换引入平坦空白伪影），再交换多频段零均值纱线残差（`h_fine` + `h_mid`）、随机翻转/旋向、调制纹理对比度并叠加平滑环境光照梯度，且采用双级 `9×9` 羽化过渡消除边界阶跃（**背景发生改变**）；
  - 所有 GT 缺陷框内部像素严格保持逐比特一致（`max_abs_diff == 0`，**疵点不变**），全幅纹理异常（如 `w * h >= 0.55` 的 *纬斜*）自动跳过保护。
- **双视图共享主干+颈部对齐监督（`YOLO27DetectionModel` + `ForegroundAlignmentBranch` + `YOLO27E2ELoss`）**：
  - 原图 $I$ 正常前向训练检测头并更新 `BatchNorm` 统计量；配对图 $\tilde{I}$ 在冻结 `BatchNorm` 滑动统计量（`_freeze_bn_running_stats`，确保推理 BN 零漂移）的前提下流经共享 **Backbone + Neck** 与无状态 `GroupNorm` 轻量分支 `ForegroundAlignmentBranch`；
  - 联合监督 $(I, \tilde{I})$ 双视图的 **IoU-Varifocal 前景显著性一致性** 与正样本锚点 **对称停止梯度余弦特征对齐损失** $\frac{1}{2}(1-\cos(\text{sg}(z^{\text{clean}}), z^{\text{pert}})) + \frac{1}{2}(1-\cos(z^{\text{clean}}, \text{sg}(z^{\text{pert}})))$，将背景不变梯度直接回传至主干与颈部全层卷积核（**泛化能力增强**）；
  - **推理完全剥离**：调用 `model.fuse()`（验证 `.pt`、预测或导出 ONNX/TensorRT）时自动执行 `self.fg_branch = None` 且不再构造 $\tilde{I}$，**推理阶段 0 额外参数、0 额外 FLOPs、0 延迟开销**。

---

## 二、模型架构与复杂度对比（`nc=19`, 输入 `640×640`）

| 模型变体 | 配置文件 | 预测尺度 (Stride) | 前景对齐辅助监督 | 训练态参数量 | 推理融合后参数量 | 融合后 FLOPs | 特点与适用场景 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO26n** (基准) | `yolo26n.yaml` | `[8, 16, 32]` | 无 | 2.511 M | 2.379 M | 5.33 B | 原生前代基准 |
| **YOLO27n** (推荐默认) | [`models/yolo27n.yaml`](models/yolo27n.yaml) | `[8, 16, 32]` | **有（训练期启用，推理剥离）** | 2.697 M | **2.491 M** | **6.55 B** | 参数贴近 YOLO26n（仅 `+0.11M`），跨底纹泛化与小目标/细长条精度最高 |
| **YOLO27n-Dual** (双尺度版) | [`models/yolo27n-dual.yaml`](models/yolo27n-dual.yaml) | `[8, 32]` | **有（训练期启用，推理剥离）** | 2.603 M | **2.450 M** | **6.42 B** | 官方 YOLO27 双尺度头，参数与 YOLO26n 几乎一致 |
| **YOLO27n-Lite** (极轻版) | [`models/yolo27n-lite.yaml`](models/yolo27n-lite.yaml) | `[8, 32]` | **有（训练期启用，推理剥离）** | 0.861 M | **0.776 M** | **3.29 B** | 工业边缘相机极限高帧率部署，兼备背景不变泛化训练 |

---

## 三、快速开始

### 1. 查看参数量、前景对齐状态与延迟对比
```bash
python train_yolo27n.py --mode profile
```

### 2. 训练模型（自动多级检索本地/同盘 `yolo26n.pt` 执行无损 Net2Wider 热启动）
> 建议将 `yolo26n.pt` 放在项目根目录、`weights/` 目录或上级/同级目录（训练器会自动检索并打印 `YOLO27 Warm-Start: transferred ...` 日志；若未找到会立即清晰提示避免静默退化为从零随机训练），也可通过 `--donor path/to/yolo26n.pt` 显式指定：

```bash
# 1) 训练默认高精度 YOLO27n（默认开启前景对齐辅助监督与背景改变-疵点不变增强）
python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --epochs 150 --batch 16 --imgsz 640

# 2) 显式指定预训练权重路径（如存放在其他目录）
python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --donor D:/tyhn/yolo/yolo26n.pt --epochs 150 --batch 16 --imgsz 640

# 3) 训练双尺度版 YOLO27n-Dual 或极轻版 YOLO27n-Lite
python train_yolo27n.py --mode train --model models/yolo27n-dual.yaml --data dataset/fabric19.yaml --epochs 150 --batch 16 --imgsz 640
python train_yolo27n.py --mode train --model models/yolo27n-lite.yaml --data dataset/fabric19.yaml --epochs 150 --batch 32 --imgsz 640
```

### 3. 验证、推理与导出（推理自动剥离前景对齐分支，支持 `nms=True` / `nms=False`）
```bash
# 默认：一对多检测头 + NMS（最高检测精度）
python train_yolo27n.py --mode val --weights runs/detect/yolo27n_fabric19/weights/best.pt --data dataset/fabric19.yaml --nms

# 可选：一对一端到端无 NMS 推理（nms=False）
python train_yolo27n.py --mode val --weights runs/detect/yolo27n_fabric19/weights/best.pt --data dataset/fabric19.yaml --no-nms

# 导出 ONNX（自动剥离前景对齐辅助分支并折叠跨尺度缩放系数）
python train_yolo27n.py --mode export --weights runs/detect/yolo27n_fabric19/weights/best.pt --format onnx
```

### 4. 运行自动化验证单测
```bash
python tests/test_yolo27n.py
```
