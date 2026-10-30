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
│   ├── metrics.py                 # AUROC/PRO/F1/mAP/NMI/ARI/cluster-acc
│   ├── ncd.py                     # 区域分析、区域合并、聚类、匈牙利匹配、新类发现
│   ├── engine.py                  # 训练/验证/EMA/阈值校准/记忆库
│   ├── inference.py               # 推理封装（切块、可视化、JSON）
│   ├── ops.py                     # 连通域/形态学/积分图（不依赖 OpenCV）
│   └── utils.py
├── tools/
│   ├── train.py  evaluate.py  infer.py  discover.py
│   ├── export_onnx.py  bench.py  prepare_dataset.py  demo.py
├── tests/                         # 26 项单元 + 端到端冒烟测试
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
# 方式 A：不改 yaml，直接用 data.root 重映射
python tools/prepare_dataset.py --root /data/fabric/source5_augmenteds

# 方式 B：生成一份路径已解析的 yaml，训练时用它
python tools/prepare_dataset.py --root /data/fabric --write-yaml dataset/fabric19.resolved.yaml
```

`prepare_dataset.py` 会输出每个 split 的图片数、标注数、空标注、缺失标注、类别直方图与不均衡比，
并可选 `--export-masks` 预计算布面掩码（训练更快、前景监督更稳）：

```bash
python tools/prepare_dataset.py --root /data/fabric --export-masks
# 然后在 configs/fabric19.yaml 里设 data.mask_dir: /data/fabric/cloth_masks
```

### 6.2 训练

```bash
python tools/train.py --config configs/fabric19.yaml \
    --set data.root=/data/fabric/source5_augmenteds

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

### 6.3 评估

```bash
python tools/evaluate.py --ckpt runs/fabric19_fabricncd_s/best.pt
python tools/evaluate.py --ckpt best.pt --save-preds preds.json   # 逐图区域与阈值
```

### 6.4 推理（单图 / 目录 / 大幅面切块）

```bash
python tools/infer.py --ckpt best.pt --source a.jpg
python tools/infer.py --ckpt best.pt --source /data/images --out outputs/pred
python tools/infer.py --ckpt best.pt --source line_scan_4096x1024.png --tile --tile-size 512 --tile-overlap 128
```

输出 `*_pred.png`（热力图 + 框 + 类别 + 分数）与 `results.json`（每张图的区域、类别、到布边距离等）。

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

---

## 8. 调参手册（按症状查）

| 症状 | 建议 |
|---|---|
| 背景/辊筒频繁误报 | 提供 `data.mask_dir` 真实掩码；提高 `data.augment.width_margin`；增大 `model.fabric_mask` 权重 |
| 布边处误报多 | 确认布面掩码质量；`model.fabric_mask.use_xy=false`（用距离而非坐标）；增加布边带权重 `loss.fabric.edge_weight` |
| 小疵点（断经、飞花）漏检 | 提高输入分辨率（`data.image_size` 640/768）；`loss.anomaly.alpha` 调高到 0.7–0.75；`eval.min_area_ratio` 调小 |
| 整幅性缺陷（纬斜）漏检 | 用 `mobilenet_v3_large` 或 `efficientnet_b0`；提高 `neck.global_channels`、`neck.dilations` |
| 正常图少量误报 | 提高 `eval.calib_sigma`（阈值更保守）；增大 `model.memory.capacity` 与训练集正常图数量 |
| 分类混淆（相似类） | 提高 `loss.supcon.weight`、降低 `heads.proto_temperature`；对稀有类重采样；增加 `loss.cls.box_shrink`（更干净的框内样本） |
| CPU 端太慢 | 用 `--set data.image_size=[320,320]`；`arch=lite` + 导出 ONNX/INT8；开启 `infer.tile` 只跑可疑区域 |
| 显存不足 | 减小 `data.batch_size`；`data.image_size=384`；`model.memory.capacity=1024` |

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
