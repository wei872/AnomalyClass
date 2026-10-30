# FabricNCD：轻量化布匹疵点检测 · 多类分类 · 新类发现

面向机织布匹（经纬纹理、布边/布中纹理不一致、布幅可变、布种繁多）的**统一轻量模型**，
参考 [AnomalyNCD (CVPR 2025)](https://arxiv.org/abs/2410.14379) 的多类异常分类框架，
把 "异常检测 + 多类分类 + 新类发现" 压进一个 **1.1 M 参数** 的网络里。

| 指标 | 数值 |
|---|---|
| 参数量（MobileNetV3-Small 主干 + 全部头） | **1.11 M**（主干 0.93 M） |
| 计算量 @512×512 | **0.62 GMACs / 1.24 GFLOPs** |
| CPU 单张延迟（2 核 Intel/AMD，PyTorch） | ≈ **100 ms**（224×224 约 23 ms） |
| 输出 | 异常热力图 + 布面掩码 + 布边带 + 19 类（可扩展）分类 + 区域框 + 嵌入 |
| 监督需求 | 有框标注可用；**无标签正常图即可冷启动**（记忆库 + 合成异常） |

```text
RGB ─► MobileNetV3-Small ─┬─ p2(1/4) ─► Gabor 经纬纹理编码 ─┐
                          ├─ p3(1/8) ──────────────────────┤
                          ├─ p4(1/16) ─────────────────────┤
                          └─ p5(1/32) ─► 全局上下文(空洞卷积+线性注意力+织造周期token)
                                                             │
                                       全局-局部融合 + FiLM ◄┘
                                                     │
                          布面/布边掩码头(1/4) ─► 布面地理嵌入(到布边距离)
                                                     │
                        ┌────────────────────────────┼───────────────────────────┐
                    异常头(1/8)                 嵌入头(64d)                原型分类器(19 类)
                        │                            │                           │
                        │                    正常特征记忆库 ◄────────────────────┘
                        ▼
        异常分数 = (1-w)·sigmoid(异常头) + w·normalize(1-记忆库相似度)
```

---

## 1. 为什么不用 YOLO？

| 维度 | YOLO 类全监督检测器 | **本方案 (FabricNCD)** |
|---|---|---|
| 新布种 / 新疵点 | 需要重新标注+训练，闭集 | 正常图即可建库冷启动；聚类发现新类 |
| 布种/光照/布幅变化 | 依赖训练分布，泛化弱 | 织造尺度归一化 + Gabor 先验 + 布幅增广 + 原型度量 |
| 前景/背景混淆（辊筒、导带、布边外背景） | 框回归容易把背景当目标 | 显式布面/背景分割 + 布边软距离，异常只在布面内统计 |
| 小/弱疵点（断经、稀缝、油点线） | 依赖 NMS 与 anchor，细长目标召回差 | 1/8 稠密异常图 + 像素级 Focal/Dice，区域由连通域给出 |
| 数据需求 | 每类数百框 | 少量框 + 大量正常图即可，支持无标签增量 |
| 可解释性 | 框 + 类别 | 热力图 + 布面掩码 + 区域 + "到布边距离" + 聚类归属 |

YOLO 的价值在"闭集、标注充分、产线稳定"的场景；布匹产线的痛点是**换布种、换幅宽、出新疵点**，
所以这里选择"异常检测 + 度量学习 + 新类发现"的开放集路线，并显式注入布匹的物理先验。

---

## 2. 关键设计（详见 [docs/DESIGN.md](docs/DESIGN.md)）

1. **经纬纹理先验**（`models/texture.py`）
   - 固定 8 组 Gabor 核（0°/45°/90°/135° × 2 波长 × 实/虚部），0°/90° 直接对应纬/经纱；
   - 加窗 FFT 估计**织造周期**（经/纬向像素周期 + 各向异性 + 置信度），既做数据侧
     **织造尺度归一化**（把不同布种的经纬间距对齐到同一尺度），又作为全局 token 注入网络；
   - 换布料种类时，模型看到的是"纹理尺度已归一化 + 相位响应"的表示，而不是某种布的具体外观。

2. **全局 + 局部双分支**（`models/glocal.py`）
   - 1/32 上做空洞卷积 + **线性注意力**（O(N)，512 输入仅 256 token）建模整幅一致性（纬斜、整幅色差）；
   - 1/8 上保留局部细节（断经、棉团、飞花只有几像素），自顶向下融合并由全局向量做 FiLM 调制。

3. **布面/背景 + 布边感知**（`models/heads.py`、`data/augment.py`）
   - 布面掩码头输出前景概率与**布边带**，异常分数、损失、区域统计都只在布面内进行 → 背景不会被判成疵点；
   - 由布面概率金字塔推出 **"到布边软距离" embedding**（可微，零额外开销），不把位置硬编码进网络；
   - 训练时做 **布幅增广**：把布面贴到随机边距的背景画布 / 复制布边延伸，布面掩码与框精确同步。
     布幅一变 → 布边位置一变 → 训练分布覆盖到，这正是"边缘位置漂移"的能力来源。

4. **泛化能力**（`data/synth.py`、`models/memory.py`）
   - 只用**正常布**建 patch 记忆库（reservoir + k-center，2048×64，约 0.5 MB），推理算最小余弦距离；
   - 6 类与真实疵点形态接近的**合成异常**（划伤、油污、cutpaste、纬斜、缺纬、飞花）做自监督；
   - 颜色抖动/模糊/噪声 + 翻转/90°/小角度旋转 + 多尺度裁剪，主干浅层冻结，避免过拟合并保留 ImageNet 语义。

5. **AnomalyNCD 三件套的工程化**（`ncd.py`）
   - *主元素二值化*：自适应阈值 + 峰值突出的连通域筛选 + 高分核心收缩框；
   - *掩码引导表征*：区域掩码平均池化 + SupCon；
   - *区域合并*：面积加权 / top-1 / max 三种策略得到图像级类别；
   - *新类发现*：余弦 k-means（轮廓系数选 k）+ 匈牙利匹配已知原型 + 相似度阈值判新类。

---

## 3. 目录结构

```text
AnomalyClass/
├── configs/fabric19.yaml          # 训练/评估/推理配置（已按布匹场景调好默认值）
├── dataset/fabric19.yaml          # 你的数据集定义（19 类）
├── fabric_ncd/
│   ├── models/
│   │   ├── backbone.py            # MobileNetV3 / EfficientNet 抽头（自动探测）+ 纯 PyTorch 备用主干
│   │   ├── texture.py             # Gabor 滤波器组、织造周期估计、纹理归一化
│   │   ├── glocal.py              # 线性注意力全局分支、全局-局部融合、FiLM、布面地理嵌入
│   │   ├── heads.py               # 异常头、布面/布边头、嵌入头、原型分类器
│   │   ├── memory.py              # 正常特征记忆库 + 紧凑度损失
│   │   └── fabric_ncd.py          # 完整模型
│   ├── data/
│   │   ├── yolo.py                # YOLO 标注读取、布面掩码、数据集/DataLoader
│   │   ├── augment.py             # 布幅增广、几何/光照增广、织造尺度归一化
│   │   ├── synth.py               # 6 类合成疵点
│   │   ├── fabric_mask.py         # 无监督布面/背景掩码估计
│   │   ├── weave.py               # numpy 版织造周期估计（部署/数据侧）
│   │   └── demo.py                # 合成演示数据
│   ├── losses.py                  # Focal/Dice/SupCon/原型 CE/紧凑度
│   ├── realtime.py                # 三线程实时流水线 (读图/前向/后处理重叠)
│   ├── metrics.py                 # AUROC/PRO/F1/mAP/NMI/ARI/cluster-acc
│   ├── ncd.py                     # 区域分析、区域合并、聚类、匈牙利匹配、新类发现
│   ├── engine.py                  # 训练/验证/EMA/阈值校准/记忆库
│   ├── inference.py               # 推理封装（切块、可视化、JSON）
│   ├── ops.py                     # 连通域/形态学/积分图（不依赖 OpenCV）
│   └── utils.py
├── tools/
│   ├── train.py  evaluate.py  infer.py  discover.py
│   ├── export_onnx.py  bench.py  prepare_dataset.py  demo.py
│   ├── resolution.py              # 按原图尺寸 + 实测经纬周期推荐输入分辨率
│   ├── rt_bench.py                # 实时性基准: 逐段耗时 + FPS + 达标判定
├── tests/                         # 36 项单元 + 端到端冒烟测试
└── docs/DESIGN.md                 # 方案细节、消融实验计划、调参手册
```

---

## 4. 安装

```bash
pip install -r requirements.txt
# 可选：ONNX 导出 + 校验
pip install onnx onnxruntime
```

要求 Python ≥ 3.9、PyTorch ≥ 2.0、torchvision ≥ 0.15。CPU 即可跑通全流程；
GPU（≥ 8 GB）上 512×512、batch 8 训练约需 6–8 GB 显存。

## 5. 先跑个演示（不需要数据集）

```bash
python tools/demo.py                # 生成合成布匹数据 -> 训练 -> 评估 -> 推理可视化
# 输出：runs/demo/{best.pt, history.csv, demo_pred_*.png, demo_inference.json}
```

演示脚本用合成经纬纹理验证整条流水线，几秒钟一轮，能直接看到热力图与告警。

---

## 6. 用真实数据训练

### 6.1 数据准备

按 `dataset/fabric19.yaml` 的约定组织：

```text
DATASET_ROOT/
  images/train/  images/val/            # 疵点图
  labels/train/  labels/val/            # YOLO txt：cls cx cy w h（归一化）或多边形
  normal/images/train/                  # 纯正常布面图（无标注，强烈建议 ≥ 疵点图数量）
  normal/images/val/
```

你的 yaml 里目前是 Windows 绝对路径（`D:/tyhn/...`），两种方式二选一：

```bash
# Windows 注意: --root 要写完整盘符 (D:/data/fabric)，只写 /data/fabric 会变成
# "当前盘符下的相对路径"，掩码会写到别的盘去 (工具现在会告警并自动改写到图像实际目录)

# 如果你的图"整幅都是布"(没有辊筒/台面/背景):
#   不要加 --export-masks, data.mask_dir 保持 null —— 布面掩码会恒为全白, 没有信息量.
#   包边/布边(左右布幅边缘)的位置信息由模型内置的 model.fabric_mask.border_prior 提供.
#   疵点(前景)掩码由 YOLO 标注实时生成, 想肉眼核对用 --export-defect-masks.

# 三种"掩码"别混淆:
#   ① 布面/前景掩码  <- --export-masks      (逐像素, 告诉你哪里是布/哪里是辊筒台面)
#   ② 疵点掩码       <- 训练时由 YOLO 标注实时生成, 不落盘 (想看图用 --export-defect-masks)
#   ③ 异常热力图     <- 模型输出, 推理/评估时生成
python tools/prepare_dataset.py --root D:/data/fabric --export-defect-masks --workers 8

# 可选：预计算布面掩码（多进程 + 按经纬周期自动降采样，550 张约 1–2 分钟）
# 会先抽 8 张判断是不是"整幅都是布"，是的话直接告警跳过（加 --force 可强导）
python tools/prepare_dataset.py --root D:/data/fabric --export-masks --workers 8
# 支持断点续跑（已存在的掩码自动跳过）与试跑：--limit 20 --mask-size 1024

# 方式 A：不改 yaml，直接用 data.root 重映射
python tools/prepare_dataset.py --root /data/fabric/source5_augmenteds

# 方式 B：生成一份路径已解析的 yaml，训练时用它
python tools/prepare_dataset.py --root /data/fabric --write-yaml dataset/fabric19.resolved.yaml
```

`prepare_dataset.py` 会输出每个 split 的图片数、标注数、空标注、缺失标注、类别直方图与不均衡比，
并可选 `--export-masks` 预计算布面掩码（训练更快、前景监督更稳）：

```bash
python tools/prepare_dataset.py --root D:/tyhn/yolo/dataset/source5_augmenteds --export-masks
# 然后在 configs/fabric19.yaml 里设 data.mask_dir: D:/tyhn/yolo/dataset/source5_augmenteds/cloth_masks
```

布面掩码估计的实测（1600×846 原图，单进程 CPU）：

| 场景 | `work_size="auto"` 落点 | auto 耗时 | 与全分辨率 IoU |
|---|---|---|---|
| 整幅是布，经纬 6 px | 800 | 0.15 s | 1.000 |
| 布幅两侧各 300 px 背景 | 800 | 0.17 s | 0.998 |
| 窄布条（布只占画面 58%） | 800 | 0.15 s | 0.998 |
| 经纬 20 px | 512 | 0.10 s | 1.000 |
| 极细经纬 3 px（不能降采样） | 不降采样 | 0.32 s | 1.000 |
| 手滑 `--mask-size 512`（纹理被平均掉） | 512 | 0.09 s | 0.99+ |

`auto` 不是简单地"缩到 512"：它先在**原始分辨率**上找到布面纹理最旺的位置量出经纬周期，
据此保证缩放后周期仍 ≥ 3 px；量不准就逐次放大 `work_size` 重试，实在量不到就退回全分辨率。
几何清理（开/闭/填洞/连通块）在回放分辨率上按同一物理尺度做，所以掩码边界不会因为降采样而变糊。

### 6.2 训练

```bash
# --root 所有工具都支持 (train/evaluate/infer/discover/rt_bench/prepare_dataset...),
# 等价于 --set data.root=...; Windows 上必须写完整盘符
python tools/train.py --config configs/fabric19.yaml \
    --root D:/tyhn/yolo/dataset/source5_augmenteds

# 常用覆盖
python tools/train.py --set train.epochs=100 --set model.backbone.arch=mobilenet_v3_large
python tools/train.py --resume runs/fabric19_fabricncd_s/last.pt
python tools/train.py --no-pretrained              # 离线环境
```

训练过程中会：

1. 用**正常图**建记忆库（无需标签，冷启动）；
2. 在**正常图**上校准绝对阈值（`pixel_threshold` / `image_threshold`），随 checkpoint 保存；
3. 每轮验证并记录像素 AUROC / PRO / F1、图像 AUROC / AP、mAP@0.5、分类 Acc / F1；
4. 按综合分保存 `best.pt`，支持 EMA、早停、AMP、断点续训。

### 6.2.1 推荐预设

> `data.image_size` 的约定是 **[高, 宽]**。默认 `image_size: auto` 会**扫描数据集里所有图的真实尺寸**，
> 取最常见的那一档、保持长宽比算出输入尺寸（短边缩到 `short_side` 为止、**不放大**），
> 因此既不会宽高写反，也能应付数据集里的尺寸混用（启动日志会打印尺寸分布）。
> 想换档位只调 `data.short_side`。

```bash
# 启动时会打印数据集的尺寸分布 + 实际输入尺寸 + 显存估算, 先看一眼这 3 行
#   原图尺寸统计 (扫描 550 张): 1012x2448 x548 (99.6%)
#   原图 1012x2448 (高x宽) -> 输入 1024x2432 (缩放 1.00 不放大, 长宽比 一致)
#   预计训练显存 (估算): 7.9 GB / 显卡 8.0 GB      <- 超过 85% 会告警
python tools/train.py --config configs/fabric19.yaml --root D:/tyhn/yolo/dataset/source5_augmenteds

# ② 轻量/低显存: 短边 640 -> [640, 1216], 1.84 GMACs, CPU 约 0.35 s/张
python tools/train.py --config configs/fabric19.yaml \
    --set data.short_side=640 --set data.batch_size=8

# ③ 超采样（小疵点特别细时才值得）: 短边 1024 -> [1024, 1952], 算力 ×1.5, bs 降到 2
python tools/train.py --config configs/fabric19.yaml \
    --set data.max_scale=1.25 --set data.batch_size=2

# ④ 只跑异常检测（先不做 19 类分类，用少量框快速验证数据质量）
python tools/train.py --config configs/fabric19.yaml \
    --set model.loss.cls.weight=0.0 --set model.loss.supcon.weight=0.0
```

### 6.2.2 输入分辨率怎么定（1600×846）

**结论：保持 1.89:1，别用正方形，也别把宽高写反。** 默认 `[832, 1600]` 就是原图分辨率（846→832 取 32 的倍数）。

正方形输入会把经纬方向拉成 1:1：经纱间距和纬纱间距在像素上变得一样，`texture.py` 的 FFT 周期估计
和 Gabor 分支就失去了"经纬各向异性"这条最有效的线索，同时还浪费 40% 的像素。

以 1600×846 的原图为参照（其它长宽比按面积同比换算）：

| 输入 (H×W) | 短边 | GMACs | CPU 延迟(2核) | 训练显存/张 | 适用 |
|---|---|---|---|---|---|
| [416, 800] | 416 | 0.79 | 0.15 s | 0.26 GB | 只有 CPU、冒烟 |
| [512, 960] | 512 | 1.16 | 0.27 s | 0.41 GB | 低配 |
| [640, 1216] | 640 | 1.84 | 0.35 s | 0.62 GB | 轻量 |
| **[832, 1600]** | **832** | **3.14** | **0.68 s** | **1.05 GB** | 1600×846 原图的默认档 |
| [1024, 1952] | 1024 | 4.75 | 1.03 s | 1.60 GB | 超采样（`max_scale: 1.25`） |

显存/算力只跟**像素总量**成正比，可自行换算：`GMACs ≈ 3.14 × (H×W)/(832×1600)`，
`显存 ≈ 0.79 GB × (H×W)/1e6 × batch`。例如 2448×1012 的原图 → 输入 [1024, 2432]（2.49 Mpx）
≈ 5.9 GMACs、1.97 GB/张，bs=4 就要 7.9 GB —— 8 GB 卡会 OOM，把 batch 降到 2 或 `short_side=768`。

（本机 CPU（2 核）实测 + 卷积乘加解析统计；显存含反向，GPU 上与 batch 近似线性。）

**怎么选**：小疵点的可检出性由"缩放后还剩几个像素"决定，织造纹理的可建模性由"缩放后经纬周期"决定。

```bash
# 先量一下你的布在原始分辨率下的经纬周期（像素）
python tools/resolution.py --source D:/tyhn/yolo/dataset/source5_augmenteds/normal/images/train --native 846x1600
```

该工具要求**缩放后周期 ≥ 4 px** 才能让 Gabor/FFT 稳定工作，周期 ≥ 3 px 时还能靠卷积纹理特征兜底。
846 的原图短边本来就不大，所以默认已经是"不缩放"；只有当原图经纬周期 < 3 px（3–4 px 的极细密布）时
才需要 `--set data.max_scale=1.25` 超采样，代价是算力 ×1.5。

**数据集尺寸不统一怎么办**：真实产线的图集常见多种尺寸（例如 1012×2448 占 48%、1112×2092 占 25%、
1052×2448 占 14%，长宽比 1.88~2.42 混在一起）。整图缩放到同一输入时，各图会被**各自**拉伸不同比例。
两种处理：

1. **（默认，够用）各向异性拉伸增广**：`augment.aspect_jitter: [0.85, 1.18]` + `aspect_prob: 0.5`。
   训练时随机独立拉伸 x/y ±15%，模型见惯各种比例，推理时那点差异就不再是分布外样本；框/掩码自动跟随，不需要改标注。
2. **（严格保比例）letterbox**：等比缩放 + 填充到目标尺寸，框坐标同步平移。零形变，但填充区浪费 10–20% 像素，
   且要改训练/评估/推理三条路径。目前**未实现**，需要时再提。

启动日志会打印尺寸分布并告警：

```
原图尺寸统计 (扫描 550 张): 1012x2448 x210 (47.7%), 1112x2092 x109 (24.8%), 1052x2448 x61 (13.9%)
WARNING 数据集里有 5 种尺寸. 训练会把每张图各自缩放到同一个输入尺寸 ...
```

**整图输入 vs 切块**：本仓库默认整图缩放（`eval.tile: false`），一次前向出全图结果，不需要拼接。
只有在"既要最高细节、又上不去分辨率"的情况下才需要切块：把 `data.augment.crop_mode` 设为 `native`
（按原像素尺度裁块训练），推理时 `--tile`。这条路径精度上限更高，但切块边界要重叠融合，部署更复杂，
因此不作为默认。

### 6.3 评估

```bash
python tools/evaluate.py --ckpt runs/fabric19_fabricncd_s/best.pt
python tools/evaluate.py --ckpt best.pt --save-preds preds.json   # 逐图区域与阈值
```

### 6.4 推理（单图 / 目录 / 大幅面切块）

```bash
python tools/infer.py --ckpt best.pt --source a.jpg
python tools/infer.py --ckpt best.pt --source /data/images --out outputs/pred
python tools/infer.py --ckpt best.pt --source scan_1600x846.png            # 默认整图, 一次前向 0.68 s (2 核 CPU)
python tools/infer.py --ckpt best.pt --source huge_4096x1024.png --tile --tile-size 832x1600 --tile-overlap 128
```

输出 `*_pred.png`（热力图 + 框 + 类别 + 分数）与 `results.json`（每张图的区域、类别、到布边距离等）。

### 6.4.1 实时推理（1660 上冲 40 FPS）

```bash
# 先量一下: 逐段耗时 + 流水线 FPS, 会自动判断是否达到 40 FPS 并给降配建议
python tools/rt_bench.py --ckpt runs/fabric19_fabricncd_s/best.pt \
    --source /data/fabric/line/*.jpg --frames 300 --fp16
```

配置里已默认打开实时开关（`infer.out_stride: 4 / probs_mul: 2 / fp16 / prep_gpu /
channels_last / weave_interval: 30 / memory_stride: 2`），常驻服务用三线程流水线：

```python
from fabric_ncd.inference import FabricPredictor
from fabric_ncd.realtime import RealtimePipeline

predictor = FabricPredictor("runs/fabric19_fabricncd_s/best.pt", device="cuda", fp16=True)
pipe = RealtimePipeline(predictor, queue_size=2).start()
for frame in camera:
    pipe.submit(frame)
for res in pipe.results():
    print(res.index, res.latency_ms, [(r["name"], r["score"]) for r in res.regions])
```

细节（预算拆分、优化清单、降配顺序、精度代价）见 [docs/REALTIME.md](docs/REALTIME.md)。

### 6.5 新类发现（无标签疵点图）

```bash
python tools/discover.py --ckpt best.pt --source /data/unlabeled_defects --out outputs/ncd
```

输出 `ncd_report.json`、每个簇的样例抠图 `cluster_samples/`，以及
`discovered_prototypes.pt`（已知原型 + 新类原型，可直接热更新分类器做在线增量）。

### 6.6 部署与压测

```bash
python tools/bench.py --size 512 --device cpu          # 参数量 / MACs / 延迟
python tools/export_onnx.py --ckpt best.pt --out model.onnx --size 512 --check
```

ONNX 输入为 `images (B,3,H,W)` 与 `weave_desc (B,6)`。`weave_desc` 由部署端用
`fabric_ncd.data.weave.estimate_period`（numpy，无 torch 依赖）计算；不需要时喂 0 即可。
计算图内**不含**连通域/区域合并等后处理，部署端移植 `fabric_ncd/ncd.py` 中
`RegionAnalyzer` 的等价逻辑（纯 numpy，无第三方依赖）。

---

## 7. 指标说明

| 指标 | 含义 | 期望 |
|---|---|---|
| `pixel_auroc` | 像素级异常排序能力 | 越高越好，> 0.95 为良好 |
| `pro` | 每区域重合率（MVTec 协议），小疵点不会被大疵点淹没 | > 0.9 为良好 |
| `pixel_f1` | 校准阈值下的像素 F1 | 与阈值策略强相关 |
| `image_auroc` / `image_ap` | 良品 vs 疵品 | > 0.95 为良好 |
| `map50` | 19 类检测 mAP@0.5（端到端：异常区 → 区域 → 分类） | 类别多时先保证 `map50_agnostic` |
| `map50_agnostic` | 类无关定位 mAP@0.5，隔离分类误差 | 用于诊断"定位"还是"分类"拖后腿 |
| `acc` / `f1_macro` | 疵点图的多类分类 | 关注稀有类 F1 |

> 说明：本方案的检测框由异常连通域 + 区域合并得到，本质是"无锚框、无 NMS 的开放集检测"，
> 因此 `map50` 会低于同数据上的全监督 YOLO——换来的能力是换布种/新疵点/无标签冷启动。
> 若产线允许，可把本模型的异常图与布面掩码作为**前置筛选**，再对候选区域接一个小分类器。

### 7.1 怎么读训练日志

每个 epoch 会额外打印两行诊断：

```
分类头 | cls 2.000 (均匀基线 ln19=2.944, 已低于基线) | 框内准确率 0.435
检测   | pixelAUROC 0.6968 | imgAUROC 0.1772 | mAP50 0.0000 | 检出率 0.00 (30 张疵点图, 平均 0.0 区域/图) | thr 0.1937
```

- **`cls` 对比 `ln(C)`**：`ln(19)=2.944` 是"瞎猜"的交叉熵。`cls` 高于它说明分类头还没学到东西，
  通常出现在 warmup 阶段（`train.warmup_epochs` 个 epoch）；低于它才算真的在学。
  `框内准确率` 的分母是框内像素，随机基线是 `1/19≈0.05`。
- **`记忆项尺度`**（校准行末尾）：正常像素到记忆库的中位余弦距离。健康值在 0.02~0.15 量级并随训练平缓变化；
  接近 1.0 说明库和当前嵌入已经脱节（余弦相似度≈0），记忆分支等于常数偏置，必须重建记忆库
  （`memory_refresh_interval: 1`）。默认每轮重建一次。
- **`检出率`**：验证集疵点图中**至少检出一个区域**的比例。它是 0 而 `pixelAUROC` 正常（>0.6）时，
  说明是**阈值/后处理**问题，不是模型问题——优先看 `thr` 与 `正常最大值`（见 `阈值校准` 那行），
  再把 `eval.calib_sigma` 调小或 `eval.normal_quantile` 调到 0.99。
- **`阈值校准`** 每 `eval.calib_interval` 轮执行一次（默认每轮，`eval.calib_batches` 张正常图），
  日志会给出 `q0.999` / `mean+kσ` / `正常最大值` 三个分量。阈值一定 ≤ 正常最大值：
  否则训练后分数分布一变尖，`mean+6σ` 会超过 1.0，任何像素都超不过阈值，验证集区域数直接归零。
- **`imgAUROC` 长期低于 0.5**：先看同一行的 `top16 / max` 两个括号值。三者都低 = 分数真的分不开；
  只有 top-k 低、peak 高 = 聚合口径问题（`eval.topk_pixels` 调小）。若三者都低但 `pixelAUROC` 正常，
  多半是**布面蒙版把疵点裁掉了**——整幅都是布的数据必须 `eval.cloth_gate: false`，
  用 `python tools/diagnose_scores.py --checkpoint best.pt` 可以对同一个权重逐图验证（不重训）。

---

## 8. 调参手册（按症状查）

| 症状 | 建议 |
|---|---|
| 背景/辊筒频繁误报 | 提供 `data.mask_dir` 真实掩码；提高 `data.augment.width_margin`；增大 `model.fabric_mask` 权重 |
| 布边处误报多 | 确认布面掩码质量；`model.fabric_mask.use_xy=false`（用距离而非坐标）；增加布边带权重 `loss.fabric.edge_weight` |
| 小疵点（断经、飞花）漏检 | 提高输入分辨率（`data.image_size` 640/768）；`loss.anomaly.alpha` 调高到 0.7–0.75；`eval.min_area_ratio` 调小 |
| 整幅性缺陷（纬斜）漏检 | 用 `mobilenet_v3_large` 或 `efficientnet_b0`；提高 `neck.global_channels`、`neck.dilations` |
| 正常图少量误报 | 提高 `eval.calib_sigma`（阈值更保守）；增大 `model.memory.capacity` 与训练集正常图数量 |
| 记忆库与当前嵌入脱节（`记忆项尺度` 接近 1.0） | 把 `train.memory_refresh_interval` 设为 1，并适当降低 `memory_warmup_batches`（24 ≈ 15 s/轮），否则记忆分支退化成常数、检出率被拖到 0 |
| 检测很弱，`pixelAUROC` 卡在 0.6~0.7 上不去 | **先别重训**：`python tools/eval_scales.py --checkpoint runs/.../best.pt --scales 0.5,0.65,0.8,1.0`。曲线随尺度明显上升 → 分辨率是瓶颈；平坦 → 分辨率不是瓶颈，回去改损失/增广/标注 |
| 验证集一个框都检不出（检出率 0） | 看 `阈值校准` 行的 `正常最大值`；降 `eval.calib_sigma`（如 3.0）、`eval.normal_quantile=0.99`、`eval.min_area_ratio` 调小 |
| 图像级 AUROC < 0.5 / 检出率忽高忽低 | 先跑 `python tools/diagnose_scores.py --checkpoint best.pt`：`topk_raw` 明显高于 `topk_gated` → 布面蒙版在裁掉疵点，设 `eval.cloth_gate: false`（整幅都是布时必须关） |
| 分类混淆（相似类） | 提高 `loss.supcon.weight`、降低 `heads.proto_temperature`；对稀有类重采样；增加 `loss.cls.box_shrink`（更干净的框内样本） |
| CPU 端太慢 | 用 `--set data.short_side=512`；`arch=lite` + 导出 ONNX/INT8；开启 `infer.tile` 只跑可疑区域 |
| 显存不足 | 减小 `data.batch_size`；`data.image_size=384`；`model.memory.capacity=1024` |

### 8.1 分辨率瓶颈怎么判定（不重训也能判）

**训练结束会自动跑**（`train.scale_sweep`，默认 `[0.5, 0.65, 0.8, 1.0]`，约 3~5 分钟），
日志里直接给出表格与结论；不需要保留 checkpoint，也不影响已保存的权重：

```
分辨率体检: 用当前权重扫 [0.5, 0.65, 0.8, 1.0]
  尺度 0.80 | 输入 800x1952 | pixelAUROC 0.7012 | imageAUROC 0.8341 | 41s
  尺度 1.00 | 输入 1012x2448 | pixelAUROC 0.7623 | imageAUROC 0.8802 | 63s
  分辨率判读: 提高尺度 pixelAUROC +0.061 -> 分辨率是瓶颈 ...
```

若想单独对某个已有的 checkpoint 复测：`python tools/eval_scales.py --checkpoint runs/.../best.pt --scales 0.5,0.8,1.0`。
想关掉自动体检就把 `train.scale_sweep` 设为 `[]`。

同一个权重在不同输入尺度上跑一遍验证集，输出 `pixelAUROC / imageAUROC / 相对耗时` 表格：

- `pixelAUROC` 随尺度明显上升（Δ > 0.05）→ 小疵点在 1/8 特征图上不足一格，值得提分辨率或改切块；
- 基本平坦 → 分辨率不是瓶颈，继续加显存只会更慢，应回去调损失/增广/标注质量。

### 8.1.1 图级分数探针（`tools/diagnose_scores.py`）

```bash
python tools/diagnose_scores.py --checkpoint runs/fabric19_fabricncd_s/best.pt --csv scores.csv
```

逐图打印 `topk_gated / topk_raw / top16_raw / max_raw / mean1pct_raw / learned_max / cloth_cov`，
并给出"疵点图均值 vs 正常图均值 vs AUROC"表格，最后自动判读属于哪一类：

- `topk_raw` 远高于 `topk_gated` → 布面蒙版在裁掉疵点 → `eval.cloth_gate: false`；
- 只有 peak 类高 → top-k 稀释 → 调小 `eval.topk_pixels`；
- 全部分不开 → 两组图可能不同源（验证集正常图与训练不是同一种布），或异常头没学到东西。

### 8.2 两个已经做过 A/B、证明**无效**的方向（别再试）

| 尝试 | 结果 | 结论 |
|---|---|---|
| 异常头改到 1/4 尺度（`heads.anomaly_stride: 4`） | pixelAUROC 0.62 → 0.40 | 多出来的只是纹理噪声；真正受限时应该整体提分辨率 |
| 用记忆库距离把"整框"正样本收缩成异常核心（`loss.anomaly.pseudo_ratio`） | 0.62 → 0.45 | 记忆库还不准时，收缩会把真实疵点像素一起丢掉 |

两者的实现都保留在代码里（默认关闭），想在自己的数据上复验可以打开对比。

---

## 9. 常见问题

**Q：没有正常图怎么办？**
A：`normal/*` 为空时会自动把"空标注文件"的图片当正常图；若一张都没有，记忆库分支自动关闭，
模型退化为纯学习式异常头（仍可用，但开放集泛化会下降）。

**Q：标注是多边形可以吗？**
A：可以，`read_yolo_labels` 同时支持 `cls cx cy w h` 与 `cls x1 y1 x2 y2 ...` 多边形，多边形会生成更精确的伪掩码。

**Q：能不能只用图像级标签（有疵/无疵）？**
A：可以，把框画满整幅或使用 `tools/` 里的弱标注策略；但定位精度会下降。推荐每类至少几十个框。

**Q：能否热更新新类？**
A：`tools/discover.py` 输出的 `discovered_prototypes.pt` 含扩展后的原型矩阵，
`PrototypeClassifier.set_prototypes()` 可在不重新训练主干的情况下扩容。

**Q：与 AnomalyNCD 官方代码的关系？**
A：本仓库是**面向布匹的独立实现**，沿用了 AnomalyNCD 的"主元素二值化 / 掩码引导表征 / 区域合并 /
新类发现"思想，但主干、纹理先验、布面-布边建模、记忆库、阈值校准与部署链路均为本项目新增。

---

## 10. 引用

```bibtex
@inproceedings{huang2025anomalyncd,
  title     = {AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios},
  author    = {Huang, Ziming and Li, Xurui and Liu, Haotian and Xue, Feng and Wang, Yuzhe and Zhou, Yu},
  booktitle = {CVPR},
  year      = {2025}
}
```

本项目代码遵循仓库原有许可；引用论文请遵守其原始许可。
