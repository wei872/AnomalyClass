# AnomalyClass — 布匹异常检测 + 缺陷分类一体化流水线

基于 [Dinomaly2](https://github.com/guojiajeremy/Dinomaly2)（全谱无监督异常检测统一框架）实现，面向 `dataset/fabric19.yaml` 中的 **19 类布匹缺陷**数据集（YOLO 格式标注）：

**先做异常检测（无监督，只用正常样本训练），再对异常区域做缺陷分类**，并输出：

- 每张图是否异常 + **与正常样本偏离的置信度** `anomaly_conf`（∈ [0,1)，等于校准阈值时为 0.5，远超阈值趋近 1）
- 每个异常区域的框 + **缺陷类别**（19 类之一，分不清时自动归为追加的 **「未知类型」**，class_id = 19）+ **分类置信度** `class_conf` + **区域异常置信度** `anomaly_conf`

整体针对 **GTX 1660 (6GB)** 做了实时性设计（见文末）。

---

## 1. 数据集约定（dataset/fabric19.yaml）

```yaml
path: D:/tyhn/yolo/dataset/source5_augmenteds/
train: .../images/train        # 带缺陷的图片（有 YOLO 标签）
val:   .../images/val
normaltrain: .../normal/images/train   # 纯正常图（无标签，训练检测器）
normalval:   .../normal/images/val     # 纯正常图（校准阈值）
nc: 19
names: ['纬线杂质', '断纬', ..., '飘絮']
# labels 与 images 同级: labels/train/*.txt, labels/val/*.txt
# 每行: <cls> <cx> <cy> <w> <h>（归一化）
```

路径解析规则：绝对路径存在则直接用；否则依次尝试相对 yaml 所在目录 / 相对 `path` / 相对 cwd；也可用 `--root` 或环境变量 `FABRIC19_ROOT` 覆盖根目录。标签目录按 `images/<split>` → `../labels/<split>` 自动推断。

## 2. 安装

```bash
pip install -r requirements.txt
# GPU 版 torch 请先装与 CUDA 匹配的 torch/torchvision，再装其余依赖
```

首次训练会自动下载 DINOv2 预训练权重到 `backbones/weights/`（需能访问 dl.fbaipublicfiles.com；下载失败可用 `--allow-random-init` 跳过，仅用于冒烟测试）。

## 3. 使用流程

```bash
# ① 训练异常检测器（无监督，只用 normaltrain 纯正常图；同时在 val 上评估 AUROC）
python scripts/train_detector.py --data-yaml dataset/fabric19.yaml \
    --backbone dinov2reg_vit_small_14 --image-size 448 --crop-size 392 \
    --total-iters 40000 --batch-size 16 --out checkpoints/detector

# ② 在 normalval 上校准阈值（img_thr / pix_thr），并评估 val 集
python scripts/calibrate.py --data-yaml dataset/fabric19.yaml \
    --det-ckpt checkpoints/detector/model_best.pth --out checkpoints/detector/thresholds.json

# ③ 训练缺陷分类器（YOLO 框裁剪 -> 19 类；自动校准 unknown_thr 与温度）
python scripts/train_classifier.py --data-yaml dataset/fabric19.yaml \
    --arch mobilenet_v3_small --epochs 30 --out checkpoints/classifier

# ④ 推理：单图 / 目录 / 视频 / 摄像头
python scripts/infer.py --image test.jpg --out runs/infer
python scripts/infer.py --dir images/ --out runs/infer
python scripts/infer.py --video fabric.mp4 --out runs/          # 输出 runs/demo.mp4
python scripts/infer.py --camera 0 --show                        # 实时摄像头

# ⑤ 实时性 benchmark（在 1660 上跑，看各环节 ms 与 FPS）
python scripts/benchmark.py --device cuda:0
python scripts/benchmark.py --device cuda:0 --half --image-size 336   # 快速档
```

## 4. 输出格式

`runs/infer/results.json`（每张图一个条目）：

```json
{
  "image": "001.png", "width": 1920, "height": 1080,
  "is_anomaly": true,
  "anomaly_score": 0.83,          // 图级异常分数 = 异常图最大值
  "anomaly_conf": 0.67,           // 与正常样本偏离的置信度 = score/(score+img_thr)
  "img_thr": 0.5,
  "latency_ms": 28.4,
  "regions": [
    {"box": [x1, y1, x2, y2], "area": 1234,
     "anomaly_conf": 0.61,        // 该区域与正常偏离的置信度
     "class_id": 5, "class_name": "包边不重叠",
     "class_conf": 0.92,           // 分类置信度
     "probs": [ ...20 维概率, 最后一维为「未知类型」概率... ]}
  ]
}
```

可视化 PNG 中：热力图 + 每个缺陷框（颜色按类别区分，「未知类型」为灰色），标签形如 `包边不重叠 0.92 | 异常 0.61`。

## 5. 方法说明

### 5.1 异常检测（Dinomaly2，无监督）
- 骨干：冻结的 **DINOv2 ViT-S/14 (reg4)**，取第 2~9 层特征；可训练 **噪声瓶颈**（Linear→Dropout→MLP→Linear）与 8 层 **线性注意力 decoder**。
- 损失：loose cosine loss（丢弃距离最大的 10% 区域，梯度降权），只用 `normaltrain` 正常图训练。
- 推理：`anomaly_map = 1 - cos(en, de)`（分组融合后平均），高斯平滑 (5×5, σ=4)；图级分数 = map 最大值。
- **阈值校准**：在 `normalval` 上跑所有正常图，`img_thr = max(正常分数)`（可调分位数/系数），`pix_thr` 同理用于像素级。
- **与正常偏离的置信度**：`anomaly_conf = score / (score + img_thr)`，单调、可解释（阈值处 = 0.5）。
- **区域定位**：`map > mask_ratio × pix_thr` → 膨胀合并 → 连通域 → 外接框（过滤小面积、加 pad）。

### 5.2 缺陷分类（19 类 + 未知类型）
- 训练样本：`images/train` 中每个 YOLO 框裁剪出来的缺陷区域（四周扩 30% 上下文），强增广；模型为轻量 **MobileNetV3-Small**（约 2.5M 参数，224 输入，单张裁剪 <1ms on 1660）。
- **「未知类型」**：推理时若区域的最大 softmax 概率 `< unknown_thr`（训练时在 val 上按分位数校准并夹在 [0.5, 0.8]），则判为 class_id=19「未知类型」，`class_conf = 1 - max_prob`；概率向量补为 20 维（末维为未知概率）。另做了温度缩放（temperature scaling）让置信度更准。
- 检测到的区域一次性 batch 送入分类器，不增加逐框串行开销。

## 6. GTX 1660 实时性

| 环节 | 模型 | 输入 | 1660 预估 |
|---|---|---|---|
| 异常检测 | DINOv2 ViT-S/14 + decoder（约 37M 参数，~55 GFLOPs） | 448×448 | **30–60 FPS**（`--half` FP16 再翻倍） |
| 快速档 | 同上 | 336×336 | **80–120 FPS** |
| 缺陷分类 | MobileNetV3-Small | 224×224/框 | **<1 ms/框**（batch 走） |
| 整体流水线 | 上述两者 | 448 | **>15 FPS，满足实时** |

实测请用 `scripts/benchmark.py`（会打印本机各环节 ms/FPS）。1660（6GB 显存）跑 448 输入显存占用 < 1.5GB，非常安全；如需更稳可 `--image-size 336` 或 `--half`。

## 7. 代码结构

```
src/
  dinomaly/        # 从 Dinomaly2 裁剪的自包含实现（只依赖 torch）
    vit_dinov2.py  # DINOv2 ViT（checkpoint key 与官方完全一致，已验证）
    uad.py         # Dinomaly 模型（冻结 encoder + 噪声瓶颈 + decoder）
    blocks.py      # decoder 用 block / 线性注意力
    losses.py      # loose cosine loss
    model.py       # 按 Dinomaly2 默认超参组装
  data/            # fabric19.yaml 解析 + YOLO 数据集读取/裁剪
  detect.py        # 检测器：训练 / 推理 / 校准 / 评估
  classify.py      # 区域分类器：训练 / 推理（含未知类型判定）
  pipeline.py      # 端到端流水线 + 可视化（中文标签）
scripts/           # train_detector / calibrate / train_classifier / infer / benchmark
configs/fabric19.yaml  # 流水线默认配置
dataset/fabric19.yaml  # 数据集路径与 19 类别（你的数据地址）
tests/smoke_test.py    # 合成数据端到端冒烟测试（CPU, 无需下载权重）
```

冒烟测试：`python tests/smoke_test.py`（自动造合成数据，跑通 训练→校准→分类→推理 全链路）。

## 8. 备注

- **中文文件名**：完全支持。Python 侧（目录扫描、YOLO 标签 `.txt`、yaml 类名、JSON 输出）原生支持 Unicode；图片读写统一走 `src/util.py` 的 `imread_unicode / imwrite_unicode`（`np.fromfile`+`cv2.imdecode` / `cv2.imencode`+`np.tofile`），绕开 Windows 上 `cv2.imread/imwrite` 不认中文路径的问题。标签文件名只需与图片同名（如 `布匹样本_001.png` ↔ `布匹样本_001.txt`）。
- 检测器评估指标（I/P-AUROC、AP、F1）在 `calibrate.py` / 训练日志中输出；像素级 GT 由 YOLO 框栅格化得到。
- 中文可视化标签需要系统有中文字体（Windows 自带黑体/微软雅黑会自动使用）；也可 `--font 路径` 指定，或把字体放进 `assets/fonts/`。
- 本实现代码来自 Dinomaly2（Apache-2.0），如用于论文/产品，请按其要求在致谢中注明。
