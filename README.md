# AnomalyClass — 布匹缺陷检测 + 分类流水线

基于 **Dinomaly2** 的异常检测器 + 轻量 CNN 分类器，**先检测、后分类**：

```
整图 448×448 ──► Dinomaly2 检测器 ──► 异常图(热力图) ──► 连通域 ──► 缺陷区域框 ──► 轻量 CNN ──► 19 类 / 未知类型
                (正常+缺陷联合训练)    1-cos(en,de)+高斯平滑   mask_thr      top_k 个区域     (letterbox 224)
```

- **联合训练**：正常图走重建损失，带框缺陷图走 margin / box 监督损失，**同一 batch 混合**（不是两个模型）。
- **不裁边缘**：`crop_size = image_size = 448`，四周暗角属于正常外观，边缘疵点也不能丢。
- **YOLO 标注一鱼三吃**：缺陷侧监督信号、分类器训练样本、验证标准。
- 目标硬件 GTX 1660 实时（448 + ViT-S 约 30~60 FPS，`--half` 再翻倍）。

---

## 目录

1. [目录结构](#目录结构)
2. [安装](#安装)
3. [数据准备](#数据准备)
4. [快速开始（5 步）](#快速开始5-步)
5. [各步骤详解](#各步骤详解)
6. [输出格式](#输出格式)
7. [调参与排障](#调参与排障)
8. [检测器优化清单（v2）](#检测器优化清单v2)
9. [与官方 Dinomaly2 对照（v3）](#与官方-dinomaly2-对照v3)
10. [测试与性能](#测试与性能)

---

## 目录结构

```
src/detect.py          Dinomaly2 检测器: 训练 / 评估 / 校准(间隙中点) / 推理 / 区域分割
src/dinomaly/          Dinomaly2 模型与损失 (含 local_cosine / margin / box_margin / masked_recon)
src/classify.py        缺陷分类器: YOLO 框裁剪训练 (letterbox 保宽高比), 未知类型校准, TTA 推理
src/pipeline.py        端到端流水线: 检测 -> 区域 -> top_k 分类 -> 热力图可视化
src/data/              YOLO 数据集读取 (Unicode 安全, 坏图跳过), 配置解析
src/util.py            日志 / 指标 / 可分离高斯平滑 / 框与掩码工具 / 中文字体
scripts/               train_detector / calibrate / infer / train_classifier / evaluate / benchmark / check_images
configs/fabric19.yaml  推荐参数一览 (脚本默认值已与之一致)
dataset/fabric19.yaml  数据集路径与 19 类类名 (改成你的实际路径)
tests/                 smoke_test.py (全链路冒烟) + test_detector_fixes.py (检测器关键修复的回归测试)
```

## 安装

```bash
pip install -r requirements.txt
# GPU: 先装对应 CUDA 版本的 torch/torchvision, 再装其余依赖
# DINOv2 backbone 预训练权重放到 backbones/weights/ (如 dinov2_vits14_reg4_pretrain.pth)
# 下载: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
```

## 数据准备

编辑 `dataset/fabric19.yaml`（Windows 路径示例见文件注释）。目录结构：

```
DATASET_ROOT/
  images/train/  images/val/     # 带疵点的图
  labels/train/  labels/val/     # YOLO txt: cls cx cy w h (归一化)
  normal/images/train/           # 纯正常图: 与 images/train 缺陷图一起参与联合训练
  normal/images/val/             # 纯正常图: 训练中评估选 model_best + 阈值校准 (必须配置!)
```

> ⚠️ `normalval` 现在同时用于**训练中选 model_best** 和**校准**。`images/val` 通常全是缺陷图，
> 没有 `normalval` 就算不出图级 AUROC，训练无法正确挑选最优权重（见 [优化清单](#检测器优化清单v2) 第 2 条）。

坏图 / 截断图在构建数据集时自动跳过（日志有计数）。可先跑 `python scripts/check_images.py` 体检。

---

## 快速开始（5 步）

```bash
# 1) 联合训练检测器
python scripts/train_detector.py --data-yaml dataset/fabric19.yaml --out checkpoints/detector

# 2) 校准阈值 (写入 img_thr / pix_thr / smooth_sigma / score_ratio)
#    --sweep: 不重训, 自动扫 平滑sigma × 图级分数取法, 采用正常/缺陷间隙最大的组合 (推荐)
python scripts/calibrate.py --data-yaml dataset/fabric19.yaml --sweep \
    --det-ckpt checkpoints/detector/model_best.pth --out checkpoints/detector/thresholds.json

# 3) 先只看检测效果 (不分类)
python scripts/infer.py --data-yaml dataset/fabric19.yaml --no-classify \
    --det-ckpt checkpoints/detector/model_best.pth --thresholds checkpoints/detector/thresholds.json \
    --dir <images/val 目录> --out runs/det_val

# 4) 训练分类器 (在 YOLO 疵点框裁剪上)
python scripts/train_classifier.py --data-yaml dataset/fabric19.yaml --out checkpoints/classifier

# 5) 端到端: 先检测区域, 再分类
python scripts/infer.py --data-yaml dataset/fabric19.yaml \
    --det-ckpt checkpoints/detector/model_best.pth --cls-ckpt checkpoints/classifier/best.pt \
    --thresholds checkpoints/detector/thresholds.json --dir <图片目录> --out runs/infer
```

**经线/纬线细长缺陷多、布匹方向固定时**，推荐这套训练参数：

```bash
python scripts/train_detector.py --data-yaml dataset/fabric19.yaml --out checkpoints/detector \
    --total-iters 10000 --no-rot90 --local-loss-w 1.0 --hard-pos-ratio 0.2 \
    --hard-normal-w 0.5 --local-loss-warmup 2000 --defect-bg-w 0.5
```

---

## 各步骤详解

### 1) 联合训练检测器 `train_detector.py`

默认 448 / `dinov2reg_vit_small_14` / 20000 iters / bs16（1660 约 11h）。断点续训：`--resume checkpoints/detector/model_last.pth`。

损失构成：

| 项 | 作用 | 开关 |
|---|---|---|
| 正常图重建（loose cosine） | 压低正常图分数 | 默认开（`--ll`） |
| 局部逐位置 cosine | 细线类信号不被全局池化稀释 | `--local-loss-w 1.0 --hard-pos-ratio 0.2` |
| 图级 margin hinge | 缺陷图分数须比同 batch 正常图高 `margin` | `--margin-w`（默认 1.0）`--margin`（默认 0.05） |
| 框级 margin hinge | 缺陷框内 token 平均距离 ≥ `margin`，直接抬高 `min_defect` | `--box-margin-w`（默认 1.0） |
| 困难正常重加权 | 当前分数偏高的正常图加大重建损失，压 `max_normal` | `--hard-normal-w 0.5` |
| 缺陷图框外重建 | 缺陷图框外是正常纹理，也参与重建 | `--defect-bg-w 0.5`（默认关） |

训练期间每 `--eval-interval`（默认 5000）iter 在 **val 缺陷图 + normalval 正常图**上评估，
按 `(I-AUROC, 相对间隙)` 字典序更新 `model_best.pth`（AUROC 饱和到 1.0 后，间隙更大的更优）；`model_last.pth` 每 `--save-interval` 无条件保存。
`--no-defect-sup` 仅消融对照用（纯无监督），正式训练保持联合。

训练配方（对齐官方 Dinomaly2，详见 [对照表](#与官方-dinomaly2-对照v3)）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--optimizer` | `stableadamw` | 官方优化器：更新量 RMS 超阈值时缩步长，坏梯度整步跳过；`adamw` 为旧行为 |
| `--lr-final-ratio` | `0.0` | 余弦退火终点 = lr×该值；0 = 退火到 0，1 = warmup 后恒定（官方默认） |
| `--no-rot90` / `--no-flip` | 关（即开增强） | 官方 2D 配方无任何增强，即两者都加；方向固定的布匹至少加 `--no-rot90` |

### 2) 校准阈值 `calibrate.py`

目标：**所有正常图判正常、所有缺陷图判异常**，不靠调参跷跷板。

- 在 `normalval` 上得 `max_normal`，在 `images/val` 缺陷图上得 `min_defect`；
- `min_defect > max_normal`（可分）→ `img_thr = (max_normal + min_defect) / 2`（间隙中点，两侧 margin 最大）；
- 重叠 → 日志告警 `separable=false`，取 `max_normal`（优先保证正常全对），并列出「各缺陷类最低图分数」定位拖后腿的类；
- `pix_thr` = 正常图像素分数的 99.99% 分位数（直方图累积，不会随图片数涨内存），用于区域分割与热力图下限；
- `thresholds.json` 同时记录训练分辨率与 `smooth_sigma`，**推理自动沿用**，不用再手工保持三处一致；
- 异常置信度 `anomaly_conf = score / (score + img_thr)`，恰在阈值上为 0.5；
- **图级分数取法** `score_ratio`：0 = 异常图最大值（默认）；>0 = 最高 ratio 比例像素的均值（官方评估用 0.01，对单点噪声峰值更稳，但会稀释很细的线状缺陷）；
- **`--sweep`**（推荐）：每张 val 图只前向一次，对同一张原始异常图扫 `smooth_sigma ∈ {4,6,8,12}` × `score_ratio ∈ {0,0.1%,0.5%,1%}`，
  按相对间隙 `gap_rel = (min_defect − max_normal)/max_normal` 排序打印整张表，自动采用最优组合并写入 `thresholds.json`，**不用重训**。
  组合是在 val 上挑的，val 图太少时 min/max 本身有噪声，结果当参考；`--score-ratio 0.01` 可手工指定。

### 3) 纯检测验证

```bash
python scripts/infer.py ... --no-classify --dir <目录> --out runs/det_val
# 每图输出 <名>_vis.png (热力图+框); results.json 含 is_anomaly / anomaly_score / img_thr / regions
```

验证正异常二分类（图级准确率 + 漏检/误报逐张列表）：

```bash
python scripts/evaluate.py --data-yaml dataset/fabric19.yaml --mode pipeline \
    --det-ckpt checkpoints/detector/model_best.pth --thresholds checkpoints/detector/thresholds.json --top-k 0
```

期望图级准确率 = 1.0、漏检/误报列表为空；不达标看日志「缺陷图 score 分位数 vs img_thr」与「漏检图 margin」，重叠则回到第 1/2 步。

### 4) 训练分类器 `train_classifier.py`

- 默认 `mobilenet_v3_small @224`，50 epochs 早停；letterbox 保宽高比 + 类别加权 + MixUp；
- 预训练权重自动下载，慢/损坏时用 `--pretrained-path` 指向本地 `.pth`；
- 区域尺寸一致性：检测框按 `box_pad=0.3` 扩展（与训练裁剪 `crop_pad=0.3` 相同）再 letterbox，**训练与推理同一套裁剪**；
- 最大 softmax 低于自动校准的 `unknown_thr` → 判「未知类型」。

验证分类器本身（GT 框直接喂分类器 → 准确率/混淆矩阵）：

```bash
python scripts/evaluate.py --data-yaml dataset/fabric19.yaml --mode gt --cls-ckpt checkpoints/classifier/best.pt
```

### 5) 端到端推理 `infer.py`

输入：`--dir` 目录 / `--image` 单图 / `--video` 视频 / `--camera 0 --show` 摄像头。常用开关：

| 参数 | 说明 |
|---|---|
| `--half` | FP16（autocast；速度提升取决于显卡，GTX 1660 无 Tensor Core，请用 `benchmark.py` 实测） |
| `--top-k 1` | 只对最异常的 1 个区域分类（默认；0=全部） |
| `--no-heatmap` | 不叠热力图 |
| `--heat-floor-ratio 1.0` | 热力图下限 = `pix_thr × 该值`，低于它的像素不上色；噪声多调大（1.3~1.6），太淡调小 |
| `--heat-region-only` | 热力图只显示在检出的区域框内 |
| `--smooth-sigma` | 一般不用设，自动沿用 `thresholds.json`；手工覆盖会提示须重校准 |

热力图用**绝对阈值**渲染：低于正常图噪声上限的像素完全透明，高于它的部分颜色与透明度随强度增大——背景不会再被按每图分位数拉伸成满屏噪点。

---

## 输出格式

`results.json`：

```json
{"image": "001.png", "is_anomaly": true, "anomaly_score": 0.83, "anomaly_conf": 0.67, "img_thr": 0.41,
 "latency_ms": 25.1,
 "regions": [{"box": [x1,y1,x2,y2], "area": 900, "anomaly_score": 0.9, "anomaly_conf": 0.69,
              "class_id": 1, "class_name": "断纬", "class_conf": 0.91, "classified": true, "probs": [...]}]}
```

- `class_id == 19` 为「未知类型」：分类器最大 softmax < `unknown_thr`，或该区域未送分类器（`classified=false`）；
- 只对最异常的 `top_k` 个区域做分类，其余区域照常输出但标未知类型。

---

## 调参与排障

### 正常/缺陷分数重叠（calibrate 告警 `separable=false`）

先看日志「各缺陷类最低图分数」定位是哪些类（常见是细长经线类），然后依次：

1. 训练加 `--local-loss-w 1.0 --hard-pos-ratio 0.2 --hard-normal-w 0.5 --defect-bg-w 0.5`，更多 iters；
2. 方向固定的布匹加 `--no-rot90`；
3. **先试 `calibrate --sweep`**：不重训，自动在 `smooth_sigma × score_ratio` 里挑间隙最大的组合；仍重叠再考虑训练时 `--smooth-sigma 8~12`；
4. 确认 `normaltrain` 覆盖全部正常变化（暗角/光照/品种）；
5. 仍只差一点：`image_size` 提到 560（14×40，合法尺寸），需重训+重校准。

### 热力图噪点多

- 显示层：`--heat-floor-ratio 1.3~1.6`，或 `--heat-region-only`；
- 分数层：`--smooth-sigma 8~12` 重训/重校准；
- 区域框层：`min_area_ratio`（默认 2e-4）调大可去掉零星小框。

### 其它

| 现象 | 处理 |
|---|---|
| 训练/校准/推理分辨率不一致 | `calibrate` 把分辨率写进 `thresholds.json`，`infer` 加载时自动核对并告警 ⚠️；默认 448 勿改 |
| Windows 训练卡住 | 已默认 `--num-workers 0`，可手动调大 |
| 中文路径/文件名 | 全部走 Unicode 安全读写，无碍 |
| 校准太慢 | `--max-images 200` 先抽样 |
| backbone 权重下载失败 | 放到 `backbones/weights/`；`--allow-random-init` 仅冒烟测试用 |
| 分类器预训练下载损坏 | 自动删缓存重试一次；仍失败用 `--pretrained-path` |

---

## 检测器优化清单（v2）

本版对检测器做了一轮审查，修复/优化如下。**前 4 项会改变训练行为，旧 checkpoint 仍可加载使用，但建议重训 + 重校准。**

| # | 问题 | 影响 | 处理 |
|---|---|---|---|
| 1 | **缺陷框掩码与图像增强不同步**：图像随机 rot90/翻转，框掩码没动 | 约 3/4 的缺陷样本框级监督落在错误位置（回归测试复现：旧代码 155/200 不重合） | 新增 `TrainAug`，同一组随机参数同时作用于图像和掩码；`crop<image` 时 RandomResizedCrop 也同步 |
| 2 | **`model_best` 选不出来**：`images/val` 全是缺陷图时 I-AUROC 恒为 NaN→0，`model_best` 停在第一次评估；且 `--eval-interval` 实际没生效（按 save_interval 评估） | 拿到的可能是 iter 1000 的权重 | 评估集加入 `normalval`；按 `(I-AUROC, 相对间隙)` 选优；`--eval-interval` 生效；缺正常图时跟随最新权重并告警 |
| 3 | **细长框 token 掩码被丢**：框下采样用 INTER_NEAREST，每 14×14 只采 1 像素 | 2px 宽的经线缺陷 token 掩码全 0，框级监督失效 | 改 INTER_AREA，只要框与 token 有重叠就算缺陷 token |
| 4 | **训练分数与推理分数不一致**：图级 hinge 用 token 网格未平滑最大值，推理/校准是上采样+高斯平滑后最大值 | 训练的 margin 并不等于校准时能拿到的间隙 | `--margin-score smooth`（默认）用与推理同款分数；`token` 为旧行为 |
| 5 | **`--smooth-sigma` 调大无效**：平滑核固定 5×5，σ=4 与 σ=8 几乎都是均值滤波 | README 旧建议的「sigma 8」实际没效果 | σ≤4 保持原版 5×5（兼容旧阈值）；σ>4 核宽 `2⌈3σ⌉+1`，大核 replicate 填充（不压暗边缘）；改用可分离高斯，大核快 ~25× |
| 6 | **sigma 需手工三处一致** | 漏改任一处阈值失效 | `thresholds.json` 记录 `smooth_sigma`，推理自动沿用；`calibrate --smooth-sigma 4` 不再被忽略 |
| 7 | **校准内存**：`pix_thr` 把所有正常图全分辨率异常图拼成一个大数组 | 千张图约数 GB，OOM | 直方图累积求 99.99% 分位数（分辨率 1e-5） |
| 8 | 方向固定的布匹被 rot90 增强削弱经/纬线敏感度 | 细长缺陷变钝 | 新增 `--no-rot90`（默认仍开，兼容旧行为，**方向固定建议关**） |
| 9 | 缺陷图只用了框内监督，框外正常纹理没用上 | 少一份正常数据 | 新增 `--defect-bg-w`（默认 0 关，建议 0.5）：框外 token（外扩 1 token）参与重建 |
| 10 | 热力图按每图分位数拉伸 + 重复推理一次 | 满屏噪点、可视化多耗一次前向 | 绝对阈值渲染；`ImageResult` 复用推理时的异常图 |

回归测试 `tests/test_detector_fixes.py` 覆盖 1/3/5/7/8/9 及评估逻辑；官方对照新增的 11~16 见下一节。

> 未做、可后续评估的方向：换 `dinov2reg_vit_base_14`（更准但 1660 上变慢）、flip-TTA 平均异常图（压噪但延迟翻倍）、优化器换 StableAdamW（与官方一致）。这些需要真实数据上的对比实验，没有盲改。

---

## 与官方 Dinomaly2 对照（v3）

逐文件对照了官方仓库（`dinomaly_2D.py` / `models/uad.py` / `utils.py` / `optimizers/StableAdamW.py` / `dinov2/`）。

**已经一致（无需改）**：Dinomaly 前向（encoder 取 2~9 层 → noisy bottleneck → 8 层 decoder → 两组 loose constraint → context-aware recenter）、
LinearAttention2 / Block / DINOv2 位置编码插值、loose loss（`p` 在 1000 iter 内线性升到 0.9，被丢弃区域梯度×0.1）、bottleneck 第一层小学习率（2e-4）、梯度裁剪 0.1、`1-cos` 异常图 + 5×5 σ=4 高斯。

**本版差异与处理**：

| # | 官方做法 | 本项目此前 | 处理 |
|---|---|---|---|
| 11 | **StableAdamW**（RMS 限幅 + NaN/Inf 跳步） | 普通 AdamW，且新增了 margin/box 等多项损失，更容易出尖峰 | 移植为 `src/dinomaly/optim.py`，默认 `--optimizer stableadamw`（与官方实现数值对照：同样的数据/种子下参数最大差 3e-8）；NaN 梯度改为整步跳过 |
| 12 | 评估图级分数 = **最高 1% 像素均值**（`max_ratio=0.01`） | 只用最大值，单个噪声像素就能抬高 `max_normal` | 新增 `score_ratio`，`calibrate --sweep` 数据驱动选取；`thresholds.json` 记录，推理自动沿用 |
| 13 | 评估时异常图先缩到 256 再平滑（σ=4，等效在 448 上的平滑强度约为我们的 1.7 倍） | 在 448 上直接用 5×5 核，平滑偏弱 → 噪点偏多 | `calibrate --sweep` 同时扫 `smooth_sigma`（4/6/8/12），不再靠猜 |
| 14 | `--lr_decay_ratio` 默认 1 → warmup 后 **恒定学习率** | 余弦退火到 0（旧实现把 `final_ratio` 当成乘数，语义和官方相反） | 调度器改为官方语义；新增 `--lr-final-ratio`，默认 0（行为不变），想完全对齐官方设 1 |
| 15 | 2D 配方 **无任何训练增强** | 固定开 rot90 + 翻转 | 新增 `--no-flip`；官方配方 = `--no-rot90 --no-flip`；布匹方向固定至少关 rot90 |
| 16 | — | **`--half` 在 CUDA 上不可用**：只把输入转成 fp16，模型权重仍是 fp32 → dtype 不匹配报错 | 改用 `torch.autocast`（权重保持 fp32，LayerNorm/softmax 自动留在 fp32）；`cudnn.benchmark` 打开（输入尺寸固定）。⚠️ 本环境无 GPU，该路径只在 CPU/bfloat16 下做了等价性测试，请在 1660 上用 `benchmark.py --half` 实测 |

**有意不跟官方的地方**：官方默认 `crop_size=392`（448 缩放后中心裁剪），本项目 `crop_size=image_size=448` 不裁边缘（四周暗角属于正常外观、边缘疵点不能丢）；
官方是纯无监督，本项目额外有缺陷监督（margin / box / 框外重建）。

**没改、需要真实数据实验再决定**：换 `dinov2reg_vit_base_14`（官方 base/large 更准，1660 上会变慢）、flip-TTA、官方的 DINOv3 backbone 路径、训练到 4 万 iter（官方 MVTec/VisA 用 40000，这里默认 20000 是按 1660 的耗时折中）。

---

## 测试与性能

```bash
python tests/smoke_test.py            # CPU 可跑, tiny_test_14 随机初始化: 训练/校准/推理/分类/可视化全链路
python tests/test_detector_fixes.py   # 检测器关键修复回归测试 (秒级): 增强同步 / 平滑核 / 校准 / StableAdamW / 调度器 / score_ratio / autocast / 扫参
python scripts/benchmark.py --data-yaml dataset/fabric19.yaml --device cuda:0   # 速度
```

448 + ViT-S 约 30~60 FPS（`--half` 再翻倍）；检测器训练 bs=16 约 2s/iter。
