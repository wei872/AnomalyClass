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
4. [一键运行（推荐）](#一键运行推荐)
5. [快速开始（5 步，手动）](#快速开始5-步)
6. [各步骤详解](#各步骤详解)
7. [输出格式](#输出格式)
8. [调参与排障](#调参与排障)
9. [检测器优化清单（v2）](#检测器优化清单v2)
10. [与官方 Dinomaly2 对照（v3）](#与官方-dinomaly2-对照v3)
11. [分类器与训练采样改进（v4）](#分类器与训练采样改进v4)
12. [训练提速与数据泄漏检查（v5）](#训练提速与数据泄漏检查v5)
13. [开源方案调研与新增模块（v6）](#开源方案调研与新增模块v6)
14. [精度再优化：检测融合与误差分析（v7）](#精度再优化检测融合与误差分析v7)
15. [测试与性能](#测试与性能)

---

## 目录结构

```
run.py                一键入口: python run.py (训练全流程, 可断点续跑) / python run.py infer --input ...
src/detect.py          Dinomaly2 检测器: 训练 / 评估 / 校准(间隙中点) / 推理 / 区域分割
src/dinomaly/          Dinomaly2 模型与损失 (含 local_cosine / margin / box_margin / masked_recon)
src/knn_detect.py      免训练 kNN 记忆库检测器 (DINOv2 中层 patch + coreset), 与 Dinomaly 同接口
src/embed_classify.py  DINOv2 冻结嵌入分类器 (余弦头 + Balanced Softmax + 特征级过采样)
src/classify.py        缺陷分类器: YOLO 框裁剪训练 (letterbox 保宽高比), 未知类型校准, TTA 推理
src/pipeline.py        端到端流水线: 检测 -> 区域 -> top_k 分类 -> 热力图可视化
src/data/              YOLO 数据集读取 (Unicode 安全, 坏图跳过), 配置解析
src/util.py            日志 / 指标 / 可分离高斯平滑 / 框与掩码工具 / 中文字体
scripts/               fuse_detectors / train_detector / calibrate / infer / train_classifier / evaluate / benchmark / check_images / check_leakage / fit_knn
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

## 一键运行（推荐）

不想记一串命令，就只用根目录的 `run.py`。原有 `scripts/*.py` 都保留，下面的一键入口只是把它们按顺序串起来。

```bash
python run.py                      # 训练全流程: 检查数据 → 检测器 → kNN → 融合 → 校准 → 分类器 → 评估
python run.py infer --input 图片或目录或视频.mp4     # 推理 (自动用训练选出的最佳检测器/阈值/分类器)
```

- 默认数据配置 `dataset/fabric19.yaml`，别处用 `--data-yaml xxx.yaml`；数据根目录用 `--root`；产物在 `checkpoints/`。
- **断点续跑**：每个步骤完成后写 `checkpoints/.done_<步骤>`，中途断了/报错后再执行同一条 `python run.py` 会自动跳过已完成的步骤。
  `--force` 全部重跑，`--from calibrate` 从某步起重跑，`--steps knn,calibrate` 只跑指定步骤，`--dry-run` 只打印将执行的命令。
- **预设** `--preset`：`full`（默认，细长缺陷配方）、`fast`（不训练 Dinomaly，只做 kNN + 分类器，几分钟出结果）、`smoke`（极小，仅验证流程）。
- 其它常用：`--iters 20000`、`--device cuda:0`、`--no-classifier`。
- 想改某一步的任意参数，用 `--set 步骤.参数=值`（可重复），步骤名 = check/detector/knn/fuse/calibrate/classifier/evaluate：
  `python run.py --set detector.enc-fp16=true --set classifier.backbone=dinov2reg_vit_base_14`
- 融合（w 在 val 上自动选，w=0/1 作为候选）后，`run.py` 会选出最佳检测器写入 `checkpoints/final.json`，`infer` 直接读取；推理时多余参数原样透传给 `scripts/infer.py`，如 `--half --top-k 3`。
- 训练结束会打印各步骤耗时、最终选用的检测器与分类器。

---

## 快速开始（5 步，手动）

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
| `--defect-frac` | `-1`（自动） | 每个 batch 缺陷图期望占比；缺陷图太少时抬到 25%，保证 margin/box 损失每步都有正样本 |
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

- 默认 `mobilenet_v3_small @224`，50 epochs 早停（按 balanced accuracy 选 best）；letterbox 保宽高比 + 加权采样均衡类别 + MixUp + 框抖动；
- 预训练权重自动下载，慢/损坏时用 `--pretrained-path` 指向本地 `.pth`；
- 区域尺寸一致性：检测框按 `box_pad=0.3` 扩展（与训练裁剪 `crop_pad=0.3` 相同）再 letterbox；训练时再对 GT 框做随机缩放/平移，模拟检测器框与人工框的差异；
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

## 分类器与训练采样改进（v4）

| # | 问题 | 影响 | 处理 |
|---|---|---|---|
| 17 | **分类器训练框 ≠ 推理框**：训练用人工 GT 框，推理用检测器热力图连通域出的框（大小/位置都不同），README 里「训练与推理同一套裁剪」只对 pad 成立 | 端到端分类准确率低于 `--mode gt` 的准确率 | 训练裁剪对 GT 框随机缩放（×0.74~1.35）+ 平移，`--box-jitter 0.3`，评估不抖 |
| 18 | **分类器的 90° 旋转增强会制造错标签**：19 类里有大量经线/纬线类（纬线杂质/断纬/纬斜 vs 经线毛球…），方向就是类别信息 | 纬向缺陷旋转后变成经向，标签噪声 | 默认关；需要时 `--rot90` 打开（翻转保留） |
| 19 | **类别均衡补偿了两次**：既用逆频率加权采样，损失里又乘了逆频率权重 | 少数类有效权重 ∝ 1/n_c²，过度补偿，多数类被压、概率校准变差 | 只保留加权采样，损失不加权；`--balance-power`（默认 1=完全均衡，0.5 折中，0=自然分布） |
| 20 | **按总体准确率选 best**：类别不均衡时多数类掩盖少数类 | 选出的 checkpoint 对稀有缺陷召回差 | 按 balanced accuracy（各类召回均值）选 best，日志同时打印 `val_acc` 和 `val_bal_acc` |
| 21 | **检测器训练 batch 里可能没有缺陷图**：均匀打乱时缺陷图占比低（如 5%，bs=16）约 44% 的 batch 一张缺陷图都没有 | 那一步 margin / box 损失等于没开 | `--defect-frac`：默认 -1=自动（自然占比 <20% 时抬到 25%，否则自然）；日志会打印自然占比和「batch 无缺陷图概率」 |

**没做（需要真实数据 / 更大改动）**：
- **难负样本挖掘**：分类器只见过真缺陷裁剪，检测器的误报区域（正常图上的热点）会被分到某一类或「未知类型」。
  更稳的做法是用训练好的检测器在 normaltrain 上跑一遍，把误报区域裁剪当作「未知类型」负样本加入分类器训练。
- 检测器 encoder 的 fp16（`autocast`）训练加速：1660 没有 Tensor Core，收益不确定，需要实测。

---

## 训练提速与数据泄漏检查（v5）

在 CPU 上对 `dinov2reg_vit_small_14 @448` 做了分段计时（bs=2）：**冻结 encoder 的前向约占一次迭代的一半**，decoder 的前向+反传约占另一半，且都以矩阵乘为主（不是胡乱的小算子）。所以真正能省时间的只有：少算（分辨率/迭代数）、算得快（融合注意力、fp16）、别让 GPU 等（数据加载）。

| # | 改动 | 预期收益 | 说明 |
|---|---|---|---|
| 22 | encoder 注意力改用 `F.scaled_dot_product_attention`（默认生效） | CPU 上 encoder 前向 **-33%**；GPU 上更明显（448 输入 N=1029，bs16 时手写版每层要物化约 400MB 的注意力矩阵） | 数值与手写 softmax 一致（最大误差 7e-8），推理也同步受益，无需任何参数 |
| 23 | `--enc-fp16`（可选）：冻结的 encoder 用 fp16 autocast | 省 encoder 耗时的约一半，并降低显存；**需要在 1660 上实测** | decoder 仍是 fp32（LinearAttention 的 eps=1e-8 在 fp16 下会变 0）；只影响训练，推理默认 fp32。先跑几百 iter，对比 loss 与不加此项是否一致 |
| 24 | DataLoader `persistent_workers=True, prefetch_factor=4` | Windows 上 spawn 启动 worker 每次都要重新 import torch/cv2（十几秒），旧版每个 epoch 结束都重启一遍，GPU 空等 | 仅 `num_workers>0` 时启用；同时 `pin_memory` 仅 CUDA 开、`non_blocking` 拷贝 |
| 25 | 训练 loss 在设备上累加，不再每步 `loss.item()` | 去掉每步一次 GPU 同步 | 日志每 100 步才同步一次 |
| 26 | `--early-stop N`（可选） | 连续 N 次评估 `model_best` 没提升就提前结束 | 配合 `--eval-interval 2000` 用；注意余弦调度按 `--total-iters` 排，提前停会少走完退火阶段 |
| 27 | `scripts/check_images.py` 直接报 `NameError`（原仓库遗留 bug：函数里用到的 import 写在了 `main()` 内部） | 工具能用了 | 导入移到模块级 |

**关于迭代数**：官方 40000 iter 是多类别（15 类混在一起）的设置；这里是单一布匹类型，decoder 要拟合的分布窄得多，通常 8000~12000 iter（配合余弦退火到 0）就接近收敛。建议先 `--total-iters 10000 --eval-interval 2500 --early-stop 2` 跑一次，看评估曲线再决定是否加长，而不是默认跑 20000（1660 约 11 小时）。

### 数据泄漏检查（可能是「精度低」的根因之一）

数据集目录叫 `source5_augmented`：如果是**先离线增强、后划分 train/val**，同一张原图的翻转/旋转/调亮副本会同时落在 train 和 val 里。后果：val 上的 AUROC、`model_best` 的选择、校准出的阈值、分类器的 `unknown_thr` 都是在「见过的图」上调的，指标虚高，换到真正没见过的新布样上就明显变差——看起来就是「精度低」。

```bash
python scripts/check_leakage.py --data-yaml dataset/fabric19.yaml --show 10
```

对每张 val 图取 8 种旋转/翻转版本的 dHash（对亮度缩放不敏感），与 train 的最近邻距离 ≤ 6（/64）判为近重复，并给出「无关图对」的距离中位数作参照。布匹纹理很均匀，dHash 区分度有限，**结果当线索用，请抽看 `--show` 打印的图对**。若比例很高（>10%），应按**原图**分组重新划分 train/val（增强只在 train 内做），再重新训练/校准。

---

## 开源方案调研与新增模块（v6）

目标：**训练快、检测准、分类准（含类别不均衡）、推理实时**。调研结论与取舍：

| 方案 | 要点 | 本项目的取舍 |
|---|---|---|
| **AnomalyDINO**（WACV'25, arXiv 2405.14529） | 冻结 DINOv2，patch 特征 + 最近邻记忆库，**免训练**；MVTec-AD 上 448 分辨率图级 AUROC 99.3（全样本），VisA 97.2 | ✅ 采用其思路做 `src/knn_detect.py`：训练 = 抽特征 + coreset，几分钟；当快速基线 / 数据变了快速重建 |
| **PatchCore**（CVPR'22） | 中层特征 + 贪心 k-center coreset | ✅ 记忆库压缩直接用 coreset（`kcenter_greedy`） |
| **EfficientAD**（WACV'24, arXiv 2303.14535） | 蒸馏的轻量 PDN 特征提取器 + 学生-教师，2 ms 延迟、600 图/秒 | ⏸ 延迟最低，但要重新训练蒸馏网络 + 自编码器，且放弃了 DINOv2 特征；你现有 1660 的 Dinomaly 已满足实时，暂不引入，若后续要上更弱的设备再评估 |
| **Dinomaly2**（现有主线） | 冻结 DINOv2 + 噪声瓶颈 + 线性注意力 decoder，重建式 | ✅ 保持为精度主线（已联合缺陷监督）|
| **长尾识别**：decoupled training（Kang'20）、Balanced Softmax（Ren'20）、logit adjustment（Menon'21） | 表示学习用自然分布，分类头再做类别均衡；训练时给 logits 加 `tau·log(先验)`，推理不加 | ✅ 采用：`src/embed_classify.py`，冻结骨干 + 余弦头 + Balanced Softmax |

### 1. 免训练 kNN 检测器（`scripts/fit_knn.py`）

```bash
python scripts/fit_knn.py --data-yaml dataset/fabric19.yaml --out checkpoints/knn --layer-sets "5,7,9;8,10;11"
python scripts/calibrate.py --det-ckpt checkpoints/knn/model_best.pth --out checkpoints/knn/thresholds.json --sweep
python scripts/infer.py --det-ckpt checkpoints/knn/model_best.pth --thresholds checkpoints/knn/thresholds.json ...
```

- **训练**：抽 ≤300 张正常图的 DINOv2 中层 patch 特征（多层各自归一化后平均）→ 贪心 coreset 压到 `--bank-size`（默认 2 万）→ 存成 `model_best.pth`。`--layer-sets` 可一次比较几组层，按 val 的 (I-AUROC, 相对间隙) 择优。
- **零改动接入**：模型 `forward` 返回与 Dinomaly 同接口的 `(en, de)`（查询特征 / 最近邻特征），`1-cos` 就是 kNN 余弦距离，所以 `calibrate / --sweep / evaluate / infer / benchmark` 全部直接可用。
- **取舍（实话）**：它的优势是**训练时间**（分钟 vs 数小时）和**换布样/加数据时几乎零成本**；推理延迟和 Dinomaly **差不多**（CPU 上 448 实测 391 ms vs 434 ms：省掉了 decoder，但多了最近邻搜索；`--bank-size` 越小越快）。精度要在你的数据上用 `fit_knn.py` 末尾打印的 I-AUROC / 间隙与 Dinomaly 比，**我没有你的数据，没法替你下结论**。它不使用缺陷标注，所以 Dinomaly 的联合监督（margin 损失）带来的增益它没有。

### 2. DINOv2 嵌入分类器（`train_classifier.py --kind embed`，新默认）

```bash
python scripts/train_classifier.py --data-yaml dataset/fabric19.yaml --out checkpoints/classifier        # embed (新默认)
python scripts/train_classifier.py --kind cnn --out checkpoints/classifier_cnn                            # 旧方案, 用来 A/B
python scripts/evaluate.py --mode gt --cls-ckpt checkpoints/classifier/best.pt                            # 两个 ckpt 各评一次
```

- **特征**：冻结 DINOv2 对裁剪（224 letterbox）取 `[CLS, 末层 patch 均值, 中层 patch 均值]`，各自 L2 归一化后拼接；骨干与检测器共用同一份权重文件，checkpoint 只存分类头（几十 KB）。
- **不均衡**：①分类头 = 余弦头 + **Balanced Softmax**（`--tau 1`；单元测试里两类 1000:20 且重叠时 balanced accuracy 0.818 → 0.919）；②稀有类**特征级过采样**：按 `--min-per-class` 给稀有类多抽增强视角（框抖动/翻转/颜色），骨干只多前向几次；③按 val balanced accuracy 选最佳；④不再同时用加权采样 + 损失加权（v4 已修）。
- **速度**：骨干只前向一次、特征缓存在内存，分类头训练秒级；旧方案每个 epoch 重读图 + 反传整个 CNN。
- **代价**：每个裁剪比 MobileNetV3-Small 重（CPU 上 4 个 224 裁剪 241 ms vs 19 ms；GPU 上每裁剪个位数毫秒）。流水线默认 `top_k=1` 只分类最热的 1 个区域，影响很小；想再省可 `--crop-size 168`（12×12 patch，约 2 倍快）。
- **分类准不准也是要在你的数据上比的**：先用 `--kind cnn` 和 `--kind embed` 各训一次，`evaluate.py --mode gt` 看每类准确率与混淆矩阵，保留更好的。

### 3. 顺手修的老 bug

- `scripts/evaluate.py` 的 `confusion_report` 用了函数里才 import 的 `np` → `--mode gt` 跑完直接 `NameError`（原仓库遗留）。
- `scripts/benchmark.py` 的 `_flops` 用了未 import 的 `torch`，被 `except` 吞掉，FLOPs 永远显示不出来。

### 没做 / 下一步

- **检测器误报区域当「未知类型」负样本**（难负样本挖掘）：用训练好的检测器在 normal 图上跑一遍，把误报区域的嵌入加入分类头作为一个背景类，比现在的「最大概率分位数」更像真正的开集拒识；嵌入分类器已具备基础（特征可缓存），需要先有检测器。
- **Dinomaly + kNN 分数融合**（两张异常图按正常分位数归一化后取加权）：没有数据没法定权重。
- **TensorRT / ONNX 导出**：若 1660 上 fp16 `--half` 仍不够快再做。

---

## 精度再优化：检测融合与误差分析（v7）

### 1. Dinomaly + kNN 融合检测器（`scripts/fuse_detectors.py`）

重建式（Dinomaly，带缺陷监督）与记忆库式（kNN）的误差模式不同：前者对「能被重建出来的缺陷」会漏，后者对「记忆库没覆盖到的正常变化」会误报。两张异常图各自按「正常图典型分数」归一化后加权：

```
fused = w · dino_map / ref_d + (1 − w) · knn_map / ref_k      # ref = 正常 val 图图级分数的均值, 使两者"正常水平"都≈1
```

```bash
python scripts/fit_knn.py --layer-sets "5,7,9" --out checkpoints/knn          # 层必须在 2~9 之间 (与 Dinomaly 目标层一致)
python scripts/fuse_detectors.py --dino-ckpt checkpoints/detector/model_best.pth \
    --knn-ckpt checkpoints/knn/model_best.pth --out checkpoints/fused/model_best.pth
python scripts/calibrate.py --det-ckpt checkpoints/fused/model_best.pth --out checkpoints/fused/thresholds.json --sweep
```

- **共用 encoder**：kNN 特征直接取 Dinomaly 前向里已算好的目标层 token，只多一次最近邻搜索（耗时 ∝ `--bank-size`），不重复跑 encoder。
- **w 自动选**：在 val 上扫 `0, .25, .5, .75, 1`，按 (I-AUROC, 相对间隙) 取最好。**w=1（纯 Dinomaly）和 w=0（纯 kNN）都是候选**，所以融合在 val 上不会比单个检测器差；若选出 w=1 会直接告诉你「融合无增益」。
- **必须重新 calibrate**：融合分数尺度与单个检测器不同，不能沿用旧的 `thresholds.json`。
- **代价**：推理 = Dinomaly + 一次最近邻搜索，延迟比纯 Dinomaly 高；在 1660 上用 `benchmark.py --half` 实测，若超出实时预算就用 `--bank-size` 调小。
- **局限**：w 是在 val 上选的，样本少时有噪声；融合的真实收益取决于两个检测器各自错在哪里，需要你的数据验证。

### 2. 误差分析输出

`evaluate.py --mode gt` 现在额外打印 **macro 平均（各类召回均值）** 和 **最常见的混淆对**（真实→预测、次数、占比）。分类准确率上不去时先看这里：

- 少数几对近义类互相混 → 考虑合并成一类、补那几类的数据，或检查标注一致性（这是精度天花板最常见的原因，换模型解决不了）；
- 某类召回极低且样本很少 → 加 `--min-per-class`、换更大的嵌入骨干（见下）；
- 很多真实类被判成「未知类型」→ 调低 `--unknown-quantile` / `--unknown-min`。

### 3. 按症状选下一步（都需要你的评估数据来判断，我没有）

| 症状 | 手段 | 代价 |
|---|---|---|
| 检测**漏报**（`min_defect` 低） | ① 融合 kNN；② 细长缺陷加 `--local-loss-w 1 --hard-pos-ratio 0.2`；③ 更大 backbone `dinov2reg_vit_base_14`；④ 更高分辨率（AnomalyDINO 的消融里 448→672：MVTec-AD 99.3→99.5，VisA 97.2→97.6，边际收益小）| ③④ 在 1660 上明显变慢 |
| 检测**误报**（`max_normal` 高） | ① 融合 kNN；② `calibrate --sweep`；③ `--hard-normal-w 0.5`；④ 跑 `check_leakage.py`（val 虚高会让阈值偏松，上线反而误报）；⑤ 补更多/更多样的正常图 | 基本无推理代价 |
| **分类**准确率低 | ① 看混淆对（上）；② 分类器换 `--backbone dinov2reg_vit_base_14`（只对 top-k 区域跑，开销小）；③ `--kind cnn` 做 A/B；④ 补稀有类数据 | ② 每个裁剪 ~3× 计算 |
| 端到端低但 gt 分类高 | 检测框质量问题: 用 `evaluate.py --mode pipeline` 看框召回；调 `--region-score-ratio` / `box_pad` | 无 |

---

## 测试与性能

```bash
python tests/smoke_test.py            # CPU 可跑, tiny_test_14 随机初始化: 训练/校准/推理/分类/可视化全链路
python tests/test_detector_fixes.py   # 回归测试 (秒级): 增强同步 / 平滑核 / 校准 / StableAdamW / 调度器 / score_ratio / autocast / 扫参 / 分类器增强 / 缺陷占比采样
python scripts/benchmark.py --data-yaml dataset/fabric19.yaml --device cuda:0   # 速度
```

448 + ViT-S 约 30~60 FPS（`--half` 再翻倍）；检测器训练 bs=16 约 2s/iter。
