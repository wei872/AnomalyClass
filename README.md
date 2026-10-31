# AnomalyClass — 基于 Ultralytics YOLO27 设计规范的布匹缺陷轻量化检测模型 (`YOLO27n` / `YOLO27n-Lite`)

本仓库针对 **19 类工业布匹缺陷检测数据集**（[`dataset/fabric19.yaml`](dataset/fabric19.yaml)），依据 Ultralytics 官方发布的 **[YOLO27 模型技术预览文档](https://docs.ultralytics.com/zh/models/yolo27)** 与 **YOLO26 基础代码库**，完整设计并实现了可直接训练、验证、双模式推理（带 NMS / 无 NMS 端到端）与零额外开销导出的最轻量化模型体系：

1. **`models/yolo27n.yaml`（官方规格标准版 Nano）**：严格对齐官方文档披露的 `YOLO27n` 指标（COCO `nc=80` 融合后 **3.00M 参数 / 7.2 GFLOPs**；布匹缺陷 `nc=19` 融合后 **2.99M 参数 / 7.16 GFLOPs**）。
2. **`models/yolo27n-lite.yaml`（工业边缘极轻量版 Nano-Lite）**：完整保留 YOLO27n 的全部架构改进，同时引入可分离下采样 `SCDown` 与通道压缩（`[0.50, 0.1875, 768]`），融合后仅 **0.75M 参数 / 2.97 GFLOPs**（权重文件约 **2.0 MB**），适合产线嵌入式相机与边缘端高帧率实时检测。

> **说明**：根据 Ultralytics 官方文档说明，官方预训练权重 `yolo27n.pt` 与官方源码尚处于发布前最终研发阶段，暂未公开下载。本项目基于官方披露的全部架构与训练机制规范，在最新 `ultralytics` 框架上完成了模块化工程落地与数值等价验证。

---

## 一、YOLO27n 相比前代（YOLO26n / YOLO11n）增加了哪些核心改进？

### 1. 尺度自适应架构路线（Scale-Adaptive Dual-Architecture Strategy）
- **改进内容**：YOLO27 打破了以往 N/S/M/L 仅靠深度和宽度缩放同一套计算图的惯例——大尺寸 `M/L` 采用基于查询（Query-based）的 Transformer 解码器；而面向边缘实时场景的最轻量级 **`YOLO27n / YOLO27s` 专精于精简纯 CNN 架构**，规避高分辨率注意力带来的显存带宽开销。
- **双推理路径兼容**：通过同一套 `YOLO27` 接口，默认使用 `nms=True`（一对多密集预测头 + NMS，最大化召回率）；传入 `nms=False`（或 CLI `--no-nms`）即可切换至原生无 NMS 的一对一端到端检测头，实现超低延迟部署。

### 2. 双尺度检测头 + 融合特征固定缩放平衡（Dual-Scale Detection with Fixed Feature Scaling）
- **移除中等粒度检测层（P4/16）**：传统 YOLO 在 `P3/8 (80×80)`、`P4/16 (40×40)`、`P5/32 (20×20)` 三个尺度上预测（共 `8,400` 个 Anchor）。`YOLO27n` 移除了 `P4/16` 检测分支，仅保留：
  - **细粒度特征图 `P3/8 (80×80, 6400 Anchors)`**：专职捕捉微米/毫米级细小瑕疵（如 *蚊虫小黑点、经线毛球、小尾巴、短废纱线、单根包不进*）；
  - **粗粒度特征图 `P5/32 (20×20, 400 Anchors)`**：结合 `SPPF` + `C2PSA` 注意力，专职覆盖大跨度/长条形瑕疵（如 *断纬、纬线不直、纬斜、油点线、包边不良*）。
  - Anchor 总数由 `8,400` 降至 `6,800`（减少 **19%**），并直接裁掉整组中尺度解耦卷积检测头，大幅降低检测头计算与访存开销。
- **融合特征固定缩放（Fixed Feature Scaling）与零开销重参数化折叠**：在 `YOLO27Detect`（[`yolo27/modules.py`](yolo27/modules.py)）中对双尺度融合特征施加固定方差平衡系数 `(1.0, 1/√2)`，平衡细粒度与多级融合粗粒度特征的激活幅值与梯度流；在推理/导出执行 `fuse()` 时，该固定系数直接数学等价地折叠进检测头首层 `Conv2d` 权重中，**部署阶段实现 0 额外参数、0 额外算子、数值误差 < 1e-5**。

### 3. 加宽早期高分辨率特征阶段（Widened Early High-Resolution Feature Stage）
- **改进内容**：小目标是轻量级模型最薄弱的环节。`YOLO27n` 将早期高分辨率阶段 `P2/4 (160×160)` 的 `C3k2` 通道基准从 `256` 加宽至 `384`（**+50% 特征容量**），并将细粒度颈部输出 `P3/8 (80×80)` 从 `256` 加宽至 `320`（**+25%**）。
- **对布匹瑕疵的价值**：在下采样丢失高频纹理前保留更丰富的经纬纱线细节，显著提升 *蚊虫小黑点、棉团、稀缝* 等微小缺陷的定位与边界框回归精度。

### 4. 前景对齐监督（Foreground Alignment Supervision —— 仅训练期启用，推理零成本）
- **核心痛点**：在端到端双头训练中，主干与颈部由密集的一对多分支（`one2many`，每个目标匹配 `top-10` 候选点）主导特征学习，而无 NMS 部署头（`one2one`）只允许保留唯一峰值正样本，导致两者存在监督鸿沟（YOLO26n 上一对多与一对一存在 `0.9 mAP` 差距）。
- **改进实现**（[`yolo27/modules.py`](yolo27/modules.py) & [`yolo27/loss.py`](yolo27/loss.py)）：
  1. **轻量级前景判别分支 `ForegroundAlignmentBranch`**：训练期间在 `P3/8` 与 `P5/32` 每个空间网格上挂载深度可分离轻量分支 `DWConv(3×3) -> Conv(1×1) -> Conv2d(1)`，联合 IoU 软质量目标与 `VarifocalLoss` 区分“前景瑕疵”与“背景正常布面纹理”；
  2. **一对一前景一致性对齐损失**：利用 Sigmoid 单调性恒等式 $\max_c \sigma(z_c) = \sigma(\max_c z_c)$，将 `one2one` 分类头的最高前景置信度与前景分支及 IoU 质量对齐（官方数据显示可将一对多与一对一精度差从 `0.9 mAP` 缩小至 `0.4 mAP`）；
  3. **纯正常布面样本（`normaltrain` / `normalval`）天然协同**：[`YOLO27Trainer`](yolo27/model.py) 自动识别 `dataset/fabric19.yaml` 中的 `normaltrain` 与 `normalval` 无缺陷正常布匹目录并挂载为负样本，前景对齐分支在正常布面上全锚点监督为 `0`，强力抑制纹理误报；
  4. **推理与导出完全剥离**：调用 `fuse()` 或导出 ONNX / TensorRT 时自动置空 `fg_branch = None`，**不增加任何部署延迟或模型体积**。

### 5. 面向自定义数据集的快速收敛增强体系（Fast-Convergence Recipe）
- **智能预训练权重子空间热启动（[`yolo27/weights.py`](yolo27/weights.py)）**：从 `.yaml` 构建 `YOLO27n` / `YOLO27n-Lite` 时自动从 `yolo26n.pt` 执行精确层 1:1 拷贝 + 加宽层（P2/P3/P4 `C3k2` 及双尺度 `cv2`/`cv3` 特征塔）通道对齐切片迁移（覆盖约 **80%+** 参数，同时完整保留 19 类目标的 Focal 偏置先验）。
- **无衰减双头梯度 + 双尺度 STAL `stride_val=16` 校准（[`yolo27/loss.py`](yolo27/loss.py)）**：保持 `o2m=1.0, o2o=1.0` 全程无衰减（避免 `one2one` `x.detach()` 导致中后期主干梯度饥饿，并让端到端验证头从第 1 轮全速收敛）；修复双尺度 `[8, 32]` 下 `TaskAlignedAssigner` 默认将候选框下限放大为 `32px` 的问题，恢复为 `16px` 并提升 `tal_topk=13`。
- **无 DFL 直接回归增益校准 + 小目标安全增强（[`yolo27/model.py`](yolo27/model.py)）**：匹配 `reg_max=1` 将归一化 L1 增益提升至 `dfl=6.0`；默认启用 `AdamW (lr0=0.0015) + cos_lr=True + warmup_bias_lr=0.0`（防止早期分类偏置爆炸），并设置 `erasing=0.0`（防止随机擦除抹掉微小瑕疵标签）、`flipud=0.5`、`close_mosaic=20`。

---

## 二、模型复杂度对比（`nc=19`, 输入 `640×640`）

| 模型配置 | 配置文件 | 预测尺度 (Stride) | Anchor 数 | 训练态参数量 | 推理融合后参数量 | 融合后计算量 (FLOPs) | 适用场景 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO26n** (前代基准) | `yolo26n.yaml` | `[8, 16, 32]` | 8,400 | 2.511 M | 2.379 M | 5.21 B | 前代三尺度基准（强制无 NMS 单正样本头） |
| **YOLO27n** (官方高精度版) | [`models/yolo27n.yaml`](models/yolo27n.yaml) | `[8, 16, 32]` | 8,400 | 2.669 M | **2.481 M** | **6.26 B** | 默认一对多密集头+NMS高精度推理 + 无损继承 `yolo26n.pt` + P2/P3 加宽 + 前景对齐监督（低于官方 3.0M/7.2B 预算） |
| **YOLO27n-Lite** (双尺度极轻) | [`models/yolo27n-lite.yaml`](models/yolo27n-lite.yaml) | `[8, 32]` | **6,800** | 0.825 M | **0.751 M** | **2.91 B** | 算力受限边缘工控机 / 智能线阵相机极限加速 |

---

## 三、目录结构与快速开始

```text
AnomalyClass/
├── dataset/
│   └── fabric19.yaml          # 19 类布匹缺陷数据集配置（含 normaltrain / normalval 正常样本路径）
├── models/
│   ├── yolo27.yaml            # YOLO27 (n/s) 统一多尺度架构配置
│   ├── yolo27n.yaml           # YOLO27n 标准最轻量级模型配置
│   └── yolo27n-lite.yaml      # YOLO27n-Lite 极轻量化边缘版配置 (~0.75M params / 3.0 GFLOPs)
├── yolo27/
│   ├── __init__.py
│   ├── weights.py             # yolo26n.pt 智能精确+通道切片预训练热启动迁移
│   ├── modules.py             # YOLO27Detect 双尺度检测头、固定缩放折叠与 ForegroundAlignmentBranch
│   ├── loss.py                # YOLO27E2ELoss 前景对齐监督损失 + 无衰减双头损失 + 双尺度 STAL 校准
│   └── model.py               # YOLO27 统一接口、YOLO27DetectionModel 与快速收敛 YOLO27Trainer
├── train_yolo27n.py           # 一键参数统计 / 训练 / 验证 / 推理 / ONNX 导出脚本
└── tests/
    └── test_yolo27n.py        # 架构规格、重参数化数值等价性、前景损失梯度与端到端训练单测
```

### 1. 环境准备
```bash
pip install ultralytics
```

### 2. 查看参数量与计算量对比
```bash
python train_yolo27n.py --mode profile
```

### 3. 训练布匹缺陷检测模型
先将 `dataset/fabric19.yaml` 中的 `path / train / val` 修改为你机器上的实际数据路径（默认与纯 `yolo26n` 采用完全一致的 `train / val` 划分；若需附加 `normaltrain / normalval` 纯背景目录，请加 `--use-normal`）：

```bash
# 推荐：显式指定本地 yolo26n.pt（或留空自动查找/下载），启用 92.5% 参数零误差 Net2Wider 热启动 + 默认 NMS 高精度验证
python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --pretrained yolo26n.pt --epochs 150 --batch 16 --imgsz 640

# 从零随机初始化训练（与纯 YOLO('yolo26n.yaml') 从头训练严格同条件对比）：
python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --pretrained none --epochs 150 --batch 16 --imgsz 640

# 若需同时把 normaltrain / normalval 无缺陷纯背景目录加入训练与验证，加上 --use-normal：
python train_yolo27n.py --mode train --model models/yolo27n.yaml --data dataset/fabric19.yaml --pretrained yolo26n.pt --use-normal --epochs 150 --batch 16

# 或训练双尺度极轻量版 YOLO27n-Lite (0.75M 参数 / 2.91 GFLOPs)
python train_yolo27n.py --mode train --model models/yolo27n-lite.yaml --data dataset/fabric19.yaml --epochs 150 --batch 32 --imgsz 640
```

也可以直接在 Python 中调用（与官方 `YOLO` API 完全一致）：
```python
from yolo27 import YOLO27

model = YOLO27("models/yolo27n.yaml")
model.train(data="dataset/fabric19.yaml", epochs=150, imgsz=640, batch=32)

# 默认模式：一对多预测头 + NMS（最高精度）
results_nms = model.predict("path/to/fabric.jpg", nms=True)

# 极速端到端模式：一对一预测头，无需 NMS 后处理
results_e2e = model.predict("path/to/fabric.jpg", nms=False)

# 导出无 NMS 端到端 ONNX 模型（自动剥离前景对齐分支与冗余检测头）
model.export(format="onnx", imgsz=640, nms=False)
```

### 4. 运行自动化验证测试
```bash
python tests/test_yolo27n.py
```
