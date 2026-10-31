# FabricNCD · Fabric19 端到端缺陷检测/定位/分类/异常评分

基于 [AnomalyNCD (CVPR 2025)](https://github.com/HUST-SLOW/AnomalyNCD) 的掩码引导区域表征学习思路，面向 `dataset/fabric19.yaml` 的 19 类布匹缺陷数据，扩展为 **FabricNCD 端到端流程**：

```text
原始图片（不需要框、标签、异常图）
  → 缺陷检测（正常/异常 + 候选框）
  → 异常定位（框 + 弱监督异常热图/叠加图）
  → 异常分类（19 类监督 / 可配置 NCD 新类发现）
  → 异常分数（图像级 + 区域级）
```

默认轻量模型：

- 检测器：`yolov8n.pt`，类别无关 defect 检测器，正常图作为背景参与训练。
- 区域分类 backbone：`dino_vits8`，对应 DINO 权重 `dino_deitsmall8_pretrain.pth`。

> 定位热图由检测框和检测分数生成，是 box-supervised weak localization，不是像素级分割掩码。若需要像素级分割，需要额外分割标注或分割模型。

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
├── normal/images/train/   # 纯正常图，无标签；作为检测器背景
└── normal/images/val/
```

原 YAML 根目录是 Windows 路径。换机器时推荐用 `--data-root` 覆盖：

```bash
python scripts/train_fabricncd.py --data-root /实际数据根目录
```

也可设置 `ANOMALYNCD_DATA_ROOT`，或直接修改 YAML 路径。预处理会检查图片可读性、标签格式、类别范围、空掩码、train/val 完全相同图片。增强数据请先按原始布卷/原图划分再增强；哈希检查只能发现完全相同文件，不能发现所有近重复增强泄漏。

## 2. 安装

建议 Python 3.10+。先按显卡/CUDA 安装匹配的 PyTorch，再安装依赖：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

pip install torch torchvision
pip install -r requirements.txt
```

无显示器的 Linux 服务器如果 OpenCV 报 `libGL.so.1`，可改用 headless OpenCV：

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

默认检测器 `yolov8n.pt` 会尝试下载 Ultralytics 预训练权重；DINO 默认下载 `dino_deitsmall8_pretrain.pth`。离线环境可分别用 `--detector-weights` 和 `--backbone-weights` 指定本地权重。

官方轻量 DINO 权重：
`https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth`

## 3. 端到端训练

### 监督模式：19 类缺陷识别

```bash
python scripts/train_fabricncd.py \
  --config configs/fabric19.yaml \
  --data-root /实际数据根目录 \
  --output-dir outputs/fabricncd_supervised
```

训练会依次完成：

1. 构建检测器数据：缺陷框转为类别无关 `defect`，正常图生成空标签作为背景；
2. 训练轻量 YOLO 检测器；
3. 生成区域裁剪/矩形掩码；
4. 训练 DINO-MGViT 19 类区域分类器；
5. 打包端到端 bundle：`detector.pt`、`classifier.pt`、`fabricncd_bundle.json`。

### 报错「train/val 有 N 个完全相同图片」怎么办

这是**数据泄漏保护**，不是程序 bug：`val` 里有图片和 `train` 里的图片逐字节相同（SHA256 一致）。
增强后再划分数据集时非常常见——同一张原图的不同增强副本被分到了两边。
带着这种重复训练，验证指标会虚高，无法反映真实泛化能力。

**先审计，看清到底是哪些文件重复：**

```bash
python scripts/audit_split.py --data-root /实际数据根目录 --report outputs/split_audit.json
# 加 --perceptual 还能检出「同一原图增强后分到两边」的近似重复（字节不同但内容几乎一样）
python scripts/audit_split.py --data-root /实际数据根目录 --perceptual
```

审计会列出每一对 `val` ↔ `train` 文件，同时统计正常图目录；发现完全重复时以退出码 1 结束，便于接入脚本。

**然后三选一：**

| 方式 | 命令 | 说明 |
| --- | --- | --- |
| 训练时自动排除（推荐，最省事） | 训练命令加 `--drop-leaked-val` | 预处理阶段把这些 `val` 图片排除，**原始数据一个字节都不改**，剔除数量写入 manifest |
| 物理隔离后再训练 | `python scripts/audit_split.py --data-root <ROOT> --quarantine <空目录> --apply` | 把重复图片及其标签移出数据集；不加 `--apply` 只预演。可用 `--restore <空目录>/quarantine.json` 完整还原 |
| 明确接受泄漏 | 训练命令加 `--allow-overlap` | 仅用于临时跑通流程，**指标不可信**，不要用于汇报结果 |

```bash
# 推荐的完整命令
python scripts/train_fabricncd.py \
  --config configs/fabric19.yaml \
  --data-root /实际数据根目录 \
  --output-dir outputs/fabricncd_supervised \
  --drop-leaked-val
```

`--drop-leaked-val` 与 `--allow-overlap` 互斥，且对检测器数据和区域裁剪**同时生效**，两条链路的 val 集合保持一致。
`scripts/prepare_data.py` 也支持这两个参数。

> **根本解法仍是按原图/布卷重新划分数据集**：先切分原始图片，再只对 `train` 做增强。
> 哈希检查只能发现完全相同的文件；旋转、裁剪、调色后的同源副本字节不同，需要用 `--perceptual` 辅助排查，
> 但它也只是启发式提示，不能替代按来源划分。

### NCD 模式：已知类 + 新类发现

编辑 `configs/fabric19_ncd.yaml`：

```yaml
discovery:
  known_class_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

然后：

```bash
python scripts/train_fabricncd.py \
  --config configs/fabric19_ncd.yaml \
  --data-root /实际数据根目录 \
  --output-dir outputs/fabricncd_ncd
```

NCD 模式下：

- 检测器必须 `class_agnostic: true`，只学习缺陷/背景和位置，不读取新类类别 ID；
- 区域分类器屏蔽新类 train 标签，val 标签只用于最终评估；
- 默认 `eval_every: 0`，不生成按 val 选出的 NCD `best.pt`，bundle 使用 `last.pt`。

常用覆盖项：

```bash
python scripts/train_fabricncd.py \
  --config configs/fabric19.yaml \
  --data-root /实际数据根目录 \
  --backbone dino_vits8 \
  --backbone-weights /path/to/dino_deitsmall8_pretrain.pth \
  --detector-weights /path/to/yolov8n.pt \
  --detector-epochs 100 --detector-image-size 640 --detector-batch-size 16 \
  --batch-size 8 --epochs 50 --device cuda
```

只验证配置和数据，不训练：

```bash
python scripts/train_fabricncd.py --config configs/fabric19.yaml --data-root /实际数据根目录 --dry-run
```

## 4. 端到端预测：只输入一张原图

```bash
python scripts/predict.py \
  --bundle outputs/fabricncd_supervised \
  --image /path/to/raw_fabric.jpg \
  --output-dir outputs/predictions/raw_fabric
```

不需要 `--boxes`、`--whole-image`、标签或异常图。输出目录包含：

- `result.json`：完整结果；
- `anomaly_heatmap.png`：异常定位热图；
- `anomaly_overlay.jpg`：原图叠加热图和检测框。

`result.json` 关键字段：

```json
{
  "is_anomaly": true,
  "prediction": "anomaly",
  "anomaly_score": 0.91,
  "detections": [{"bbox_xyxy": [10, 20, 100, 80], "detector_confidence": 0.91}],
  "regions": [
    {
      "bbox_xyxy": [10, 20, 100, 80],
      "prediction_id": 3,
      "prediction_name": "纬线不直",
      "classification_confidence": 0.87,
      "anomaly_score": 0.91
    }
  ],
  "image_prediction_ids": [3],
  "image_prediction_names": ["纬线不直"],
  "localization": {"heatmap": "...", "overlay": "..."}
}
```

正常图示例：

```json
{
  "is_anomaly": false,
  "prediction": "normal",
  "anomaly_score": 0.0,
  "regions": [],
  "image_prediction_ids": []
}
```

阈值：

- `--conf-threshold`：检测器候选框置信度阈值；
- `--anomaly-threshold`：图像级异常分数阈值；
- `--image-threshold`：区域分类合并后的图像多标签阈值；
- `--top1`：每图只输出最高概率缺陷类；
- `--novel-only`：NCD bundle 专用，明确只在新增类中分类。

异常分数当前来自检测器 defect confidence；区域分类置信度单独输出。NCD 区域结果还会输出 `novel_probability`。

## 5. 配置

`configs/fabric19.yaml` 是监督默认配置；`configs/fabric19_ncd.yaml` 继承它。关键项：

```yaml
model:
  backbone: dino_vits8
  pretrained_path: null

detector:
  model: yolov8n.pt
  image_size: 640
  epochs: 100
  batch_size: 16
  class_agnostic: true
  conf_threshold: 0.25
  iou_threshold: 0.45
  anomaly_threshold: 0.25
```

区域分类仍使用 MGViT、双视图增强、多头原型、SupCon/自蒸馏/InfoNCE/ME-Max 等思路。多 head 用累计训练损失选择，不用 val 标签挑 NCD head。checkpoint 使用 `torch.load(weights_only=True)`。

## 6. 仅区域分类/评估的旧接口

如果已经有外部检测框，只想做区域分类，仍可使用：

```bash
python scripts/prepare_data.py --data-root /实际数据根目录
python scripts/train.py --config configs/fabric19.yaml
python scripts/evaluate.py --checkpoint outputs/fabric19_supervised/best.pt
python scripts/predict_region.py --checkpoint outputs/fabric19_supervised/best.pt \
  --image /path/to/fabric.jpg --boxes /path/to/boxes.txt
```

这不是端到端入口；端到端入口是 `train_fabricncd.py` 和 `predict.py`。

## 7. 测试与当前验证状态

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check anomalyncd scripts tests
python -m compileall -q anomalyncd scripts
```

当前验证：`pytest -q` 为 **40 passed**，`ruff check` 通过。合成数据 CPU 测试覆盖：默认轻量配置、检测器数据构建/正常背景/NCD 类别隔离、端到端四阶段输出、正常图不强行分类、中文路径、Windows 根目录重定位、标签/泄漏检查、split 审计与隔离还原、感知哈希近似重复、MGViT 梯度、监督/NCD 训练评估、中断续训、安全 checkpoint、位置编码插值、CLI 和推理输入保护。另已完成随机初始化 YOLOv8n + tiny MGViT 的端到端训练/打包/原图预测 CPU 冒烟，并在人为注入 13 张重复图片的数据上验证了 `--drop-leaked-val`：检测器与区域两条链路均剔除 13 张、train/val 哈希交集为 0、原始文件未改动，训练与四阶段预测正常完成。

**本仓库当前没有真实 Fabric19 图片，也没有在真实数据/GPU/官方预训练权重上完成训练验证。** 合成测试通过只说明流程和张量/标签逻辑可运行，不代表真实布匹缺陷检测/分类准确率。

## 8. Attribution

参考上游提交 `9351c5daab246ff5db6841fde1689000d4f706f0`。代码来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `third_party/`：AnomalyNCD、DINO、SimGCD、Ultralytics。Ultralytics 为 AGPL-3.0；商业闭源使用请自行确认许可合规。数据与预训练权重不随仓库分发，仍受各自原始许可约束。
