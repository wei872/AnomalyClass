# AnomalyClass · Fabric19 适配版

基于 [AnomalyNCD (CVPR 2025)](https://github.com/HUST-SLOW/AnomalyNCD) 的掩码引导区域表征学习思路，适配 `dataset/fabric19.yaml` 中的 **19 类布匹缺陷 YOLO 检测数据**。

支持两种模式：

| 模式 | 配置 | 训练标签 | 用途 |
|---|---|---|---|
| `supervised` | `configs/fabric19.yaml` | 使用全部 19 类 train 标签 | 已有类别识别，默认模式 |
| `discovery` | `configs/fabric19_ncd.yaml` | 已知类用标签；新类标签屏蔽为 `-100` | 可配置的新类发现/NCD |

> **重要边界**：本仓库使用 YOLO 框生成近似矩形掩码和缺陷区域裁剪，不包含原论文的 MEBin/异常热图流程，也不是像素级分割或端到端检测器。它复用了 MGViT、对比/自蒸馏学习和面积加权图像合并等思路，但**不承诺复现论文指标**。正常图目录只统计审计，不作为第 20 类参与区域分类。

## 1. 数据与类别

保留 `dataset/fabric19.yaml` 的类别名和顺序，ID 从 0 开始：

| ID | 类别 | ID | 类别 | ID | 类别 | ID | 类别 |
|---:|---|---:|---|---:|---|---:|---|
| 0 | 纬线杂质 | 5 | 捻接器接头 | 10 | 纬斜 | 15 | 小布头 |
| 1 | 断纬 | 6 | 小尾巴 | 11 | 短废纱线 | 16 | 稀缝 |
| 2 | 多股线 | 7 | 经线毛球 | 12 | 单根包不进 | 17 | 油点线 |
| 3 | 纬线不直 | 8 | 包边不良 | 13 | 棉团 | 18 | 飘絮 |
| 4 | 包边不重叠 | 9 | 手工接头 | 14 | 蚊虫小黑点 | | |

期望目录：

```text
DATASET_ROOT/
├── images/train/          # 缺陷图，可含中文/多级子目录
├── images/val/
├── labels/train/          # 与图片相对路径同名的 .txt
├── labels/val/            # 每行: class_id cx cy w h，坐标归一化到 0~1
├── normal/images/train/   # 纯正常图，无标签；只统计，不训练区域分类
└── normal/images/val/
```

原 YAML 的根目录是 Windows 路径 `D:/tyhn/yolo/dataset/source5_augmenteds/`。换机器时**不必改类别顺序**，推荐在预处理时覆盖根目录：

```bash
python scripts/prepare_data.py --data-root /实际数据根目录
```

也可设置环境变量 `ANOMALYNCD_DATA_ROOT`，或直接修改 YAML 的 `path/train/val/normaltrain/normalval`。自定义绝对路径若不在原 `path` 下，请改成相对数据根目录的路径，避免静默读错数据。

预处理会检查图片可读性、标签格式、类别范围、空掩码、train/val 完全相同图片，并生成：

```text
data/processed/fabric19/
├── manifest.json          # 样本、原图 ID、框、掩码面积、数据指纹
├── images/train|val/      # 每个目标框一个方形上下文裁剪
└── masks/train|val/       # 仅当前目标框为白色，避免多类混在一个样本里
```

默认拒绝 train/val 完全重复图片。增强数据请先按**原始布卷/原图**划分再增强；哈希检查只能发现完全相同文件，不能发现所有近重复增强泄漏。

## 2. 安装

建议 Python 3.10+。先按显卡/CUDA 安装匹配的 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 改用: .\.venv\Scripts\Activate.ps1

# 示例：请按本机 CUDA 选择官方安装命令，不要盲目照抄 CUDA 版本
pip install torch torchvision
pip install -r requirements.txt
```

CPU 可运行小型测试；默认 ViT-B/8、224 输入、双视图训练建议用 GPU。默认 `batch_size=8`、`accumulation_steps=2`，等效 batch 16；显存不足可降 batch、提高累积步数，或改用 `dino_vits8`。

### DINO 预训练权重

默认 `dino_vitb8` 会尝试从官方地址下载 DINOv1 ViT-B/8 backbone 权重。离线环境请先下载，再传本地路径：

```bash
python scripts/train.py --backbone-weights /path/to/dino_vitbase8_pretrain.pth
```

官方 ViT-B/8 权重地址：
`https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth`

也支持 `dino_vitb16`、`dino_vits8`、`dino_vits16`。小模型官方文件名是 `dino_deitsmall8_pretrain.pth` / `dino_deitsmall16_pretrain.pth`。非 224 输入会对 checkpoint 的方形位置编码做插值，CLS 位置保留；其余权重严格加载，不静默跳过缺失键。

`--backbone-weights` 需要的是 **DINO 原始 backbone state_dict**，不是原版 AnomalyNCD 分类 checkpoint，也不是本项目训练后的 `best.pt/last.pt`。后两者用于 `--resume`、`evaluate.py` 或 `predict.py`。

## 3. 预处理与检查

在仓库根目录运行：

```bash
# 只扫描校验，不写裁剪
python scripts/prepare_data.py --data-root /实际数据根目录 --check-only

# 生成监督/NCD 共用的区域裁剪
python scripts/prepare_data.py --data-root /实际数据根目录

# 已确认要替换旧预处理目录时
python scripts/prepare_data.py --data-root /实际数据根目录 --overwrite
```

Windows 示例：

```powershell
python scripts/prepare_data.py --data-root D:/tyhn/yolo/dataset/source5_augmenteds
```

缺标签默认报错。确认某些图片是背景图并想跳过时，可加 `--allow-missing-labels`；纯正常图更推荐放到 `normal/images`。输出目录不能包含或位于输入图片/标签目录内，避免覆盖原始数据。

## 4. 监督模式：19 类识别

```bash
# 先验证配置、类别、数据和输入张量，不下载权重、不训练
python scripts/train.py --dry-run

# 正式训练；默认读取 configs/fabric19.yaml
python scripts/train.py --config configs/fabric19.yaml
```

常用覆盖项：

```bash
python scripts/train.py \
  --config configs/fabric19.yaml \
  --prepared-root data/processed/fabric19 \
  --output-dir outputs/fabric19_supervised \
  --backbone dino_vitb8 \
  --backbone-weights /path/to/dino_vitbase8_pretrain.pth \
  --batch-size 8 --num-workers 4 --epochs 50 --device cuda
```

监督模式每个 epoch 可观察 val 指标，并按 val region macro-F1 保存 `best.pt`；`last.pt` 始终保存。中断后续训必须保持原实验目录和模型/数据/training/loss 配置一致：

```bash
python scripts/train.py --config configs/fabric19.yaml --resume outputs/fabric19_supervised/last.pt
```

## 5. NCD 模式：已知类 + 新类发现

编辑 `configs/fabric19_ncd.yaml`：

```yaml
discovery:
  known_class_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

这只是示例：0–9 为已知类，10–18 为新类。请按业务修改。要求：

- 已知类 ID 无重复、在 0–18 内，至少留一个新类；
- train 中每个已知类和新类都必须有样本；
- 新类训练标签在 Dataset 返回前屏蔽，损失和采样器只使用“已知/未知”成员关系，不使用新类 ID；
- val 标签只用于评估，不用于补样本、训练或选择 NCD checkpoint；
- NCD 默认 `eval_every: 0`，只在训练结束后评估，使用 `last.pt`，不生成按 val 选出的 `best.pt`。

```bash
python scripts/train.py --config configs/fabric19_ncd.yaml --dry-run
python scripts/train.py --config configs/fabric19_ncd.yaml
# 或命令行覆盖已知类
python scripts/train.py --config configs/fabric19_ncd.yaml --known-classes 0,1,2,3,4,5,6,7,8,9
```

NCD 输出概率列顺序是：**已知类 ID 按配置顺序 + 其余 YOLO ID 升序**。监督模式则是 0–18。所有结果文件都记录 `class_order` 和类别名，避免把输出索引误当原始 ID。

主评估指标在全部 logits 上计算，不用真实标签把候选限制到 known/novel 子集。报告中的 `region_novel_closed_set_diagnostic` 明确假设输入已经是新类，仅作诊断，不是部署指标。

## 6. 评估

```bash
# 监督模式
python scripts/evaluate.py \
  --checkpoint outputs/fabric19_supervised/best.pt \
  --prepared-root data/processed/fabric19 \
  --output outputs/fabric19_supervised/evaluation

# NCD 模式
python scripts/evaluate.py \
  --checkpoint outputs/fabric19_ncd/last.pt \
  --output outputs/fabric19_ncd/evaluation
```

评估会校验预处理数据指纹与 checkpoint 一致。可用 `--batch-size`、`--num-workers`、`--device` 调整运行资源，用 `--image-threshold 0.2` 启用图像多标签，或 `--top1` 每图只输出最高概率类。

输出：

- `metrics.json`：区域 accuracy/balanced accuracy/macro-F1/NMI/ARI/Hungarian、图像 micro/macro/samples-F1、类别顺序、决策规则；
- `region_predictions.csv`：每个框的真实/预测原始 ID、名称、置信度、概率；
- `image_predictions.csv`：按稳定 `image_id` 合并后的多标签真实/预测结果。

图像合并使用 `sqrt(mask_area) / area_temperature` 加权区域概率。默认 `image_threshold: null` 为 top-1；一张图有多个缺陷时，应设置阈值并查看多标签指标，而不是只看 top-1。

## 7. 单图预测

本模型做的是**缺陷候选区域分类**，需要外部检测器/人工框，或明确使用整图兜底。它不会自动判断正常/异常，也没有 bundled detector。

### 使用 YOLO 预测 txt

格式为 `cls cx cy w h [conf]`，坐标归一化。输入 `cls` 会被忽略，类别由本模型重新预测：

```bash
python scripts/predict.py \
  --checkpoint outputs/fabric19_supervised/best.pt \
  --image /path/to/fabric.jpg \
  --boxes /path/to/fabric.txt \
  --conf-threshold 0.25 \
  --output outputs/prediction.json \
  --save-crops outputs/prediction_crops
```

### 使用像素 XYXY JSON

```json
{"boxes": [[100, 50, 260, 180], [400, 220, 520, 330]], "scores": [0.92, 0.81]}
```

```bash
python scripts/predict.py --checkpoint outputs/fabric19_supervised/best.pt \
  --image /path/to/fabric.jpg --boxes /path/to/boxes.json --top1
```

### 无框整图兜底

```bash
python scripts/predict.py --checkpoint outputs/fabric19_supervised/best.pt \
  --image /path/to/known_defective.jpg --whole-image
```

`--whole-image` **假设图中确有缺陷**，不适合拿正常图测试“正常识别”。NCD checkpoint 可加 `--novel-only`，但这也明确假设输入是新类，只在新增类中归一化分类。`--save-crops` 要求新目录或空目录，避免覆盖已有裁剪。

## 8. 配置与实现要点

- `configs/fabric19.yaml` 是监督默认配置；`fabric19_ncd.yaml` 通过 `extends` 继承它。
- 数据路径、裁剪 padding/min crop、模型、训练、损失、推理阈值和实验输出均可配置。
- MGViT 基于 DINOv1，默认最后 9 层使用 mask-guided CLS attention，只微调最后一个 block；支持 batch=1、极小掩码和空掩码回退。
- 双视图同步增强，默认不做 90° 旋转，避免交换经纬向语义。
- 监督损失：多头 CE + SupCon；NCD：已知类 CE/SupCon + 新类跨视图自蒸馏/InfoNCE/ME-Max。
- 多 head 用累计**训练**损失选择，不用 val 标签挑 NCD head。
- checkpoint 使用 `torch.load(weights_only=True)`，保存模型、优化器、调度器、AMP scaler、RNG、类别映射和数据指纹。
- 训练/评估/预测输出均保留原始 YOLO 类别 ID 和中文名称。

## 9. 测试与当前验证状态

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check anomalyncd scripts tests
python -m compileall -q anomalyncd scripts
```

当前验证：`pytest -q` 为 **25 passed**，`ruff check` 和 `compileall` 通过；另用随机权重的 `dino_vits16` 结构验证了本地严格加载、224→64 位置编码插值和有限前向输出。合成数据 CPU 测试覆盖：中文/嵌套路径、Windows 根目录重定位、标签/泄漏检查、裁剪掩码、MGViT 小批量梯度、NCD 标签屏蔽、监督/NCD 训练评估预测、中断续训、安全 checkpoint、位置编码插值、CLI 端到端和推理输入保护。

**本仓库当前没有真实 Fabric19 图片，也没有在真实数据/GPU/官方 DINO 权重上完成训练验证。** 合成测试通过只说明流程和张量/标签逻辑可运行，不代表布匹缺陷准确率，更不能作为论文复现结果。

## 10. Attribution

参考上游提交 `9351c5daab246ff5db6841fde1689000d4f706f0`。代码来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `third_party/`：AnomalyNCD、DINO、SimGCD。数据与预训练权重不随仓库分发，仍受各自原始许可约束。
