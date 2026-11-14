# AnomalyClass — 布匹缺陷检测 + 分类流水线

给一张原始布匹图，先判断**正常 / 异常**，异常图再由**二次判断**结合整图与异常区域给出结论：**19 类缺陷 / 未知类型（没见过的异常）**，或推翻为正常。

```
整图 448×448 ─► 检测器(Dinomaly2+kNN) ─► 正常 ──────────────────────────────────────► 直接输出「正常」
                      │
                      └► 异常: 热力图 → 连通域 → 异常区域框
                                               │
                              二次判断 (整图 + 区域 + 几何 一起看, 学出来的结论)
                                               ├─ 像正常纹理                 → 推翻, 输出「正常」
                                               ├─ 像某个已知缺陷             → 19 类之一
                                               └─ 不像任何已知类 且 区域大、异常重 → 「未知类型」
```

- **检测器**：Dinomaly2 重建式异常检测，**正常图 + 带框缺陷图联合训练**（同一 batch 混合，不是两个模型）；训练末尾自动再建一个免训练的 kNN 记忆库并与之融合，权重在 val 上自动选。
- **二次判断分类器（`--kind judge`，默认）**：Dinomaly2 说异常**不算数**，要再看一遍：每个异常区域的特征 = 区域裁剪 + **整图上下文** + **区域内 / 区域外** token 特征 + 面积/宽高比，一个余弦头学出「19 类缺陷 + 正常」的结论；「未知类型」= **新颖度**（没见过）**且严重度**（Dinomaly2 视角：图级分数和区域面积都够大）。新颖度用留一类（leave-one-class-out）学出来，不需要真实未知样本。原理见 §7。
- **不裁边**：训练/校准/推理统一 448，`crop_size = image_size`，四周暗角属于正常外观，边缘疵点也不会丢。
- **YOLO 标注一鱼三吃**：缺陷侧监督信号、分类器训练样本、验证标准。
- 目标硬件 GTX 1660 实时（448 + ViT-S，纯 Dinomaly 约 30~60 FPS，未在 1660 实测，请用 `benchmark.py` 验证）。

## 命令总览

训练 3 条 + 预测 1 条（预测有「完整」和「纯正异常检测」两种模式，**同一个脚本，加 `--no-classify` 就是纯检测**）：

```bash
python scripts/train_detector.py                  # ① 训练正/异常检测器
python scripts/calibrate.py                       # ② 校准阈值
python scripts/train_classifier.py                # ③ 训练缺陷分类器 (只做纯检测可跳过)
python scripts/infer.py --input <图/目录/视频>                 # ④ 完整预测: 正/异常检测 + 缺陷分类
python scripts/infer.py --input <图/目录/视频> --no-classify   # ④ 纯正异常检测 (只要 ①② 的权重)
```

## 目录

1. [安装](#1-安装)
2. [数据准备](#2-数据准备)
3. [四条命令](#3-四条命令)
4. [两种预测方式：完整预测 / 纯正异常检测](#4-两种预测方式完整预测--纯正异常检测)
5. [输出格式](#5-输出格式)
6. [验证与辅助命令](#6-验证与辅助命令)
7. [原理简述](#7-原理简述)
8. [调参与排障](#8-调参与排障)
9. [项目结构](#9-项目结构)
10. [测试](#10-测试)

---

## 1. 安装

```bash
pip install -r requirements.txt
# GPU: 先装对应 CUDA 版本的 torch/torchvision, 再装其余依赖
```

DINOv2（带 register）预训练权重**首次运行会自动下载**到 `backbones/weights/`（检测器、kNN、分类器共用同一份）。
机器不能联网时，手动下载后放到该目录即可：

```
backbones/weights/dinov2_vits14_reg4_pretrain.pth
# 下载: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
# base 版: dinov2_vitb14_reg4_pretrain.pth (对应 --backbone dinov2reg_vit_base_14)
```

`--weights-dir` 可指定其它目录；`--allow-random-init` 只用于冒烟测试，不要用于真实训练。

## 2. 数据准备

编辑 `dataset/fabric19.yaml`（五个路径 + 19 个类名），或不改文件、运行时加 `--root <数据根目录>`（也可设环境变量 `FABRIC19_ROOT`）。目录结构：

```
DATASET_ROOT/
  images/train/   images/val/      # 带疵点的图
  labels/train/   labels/val/      # YOLO txt: cls cx cy w h (归一化)
  normal/images/train/             # 纯正常图 (无标签): 与缺陷图一起参与联合训练
  normal/images/val/               # 纯正常图: 训练中选最优权重 + 校准阈值 (必须配置!)
```

> ⚠️ `normalval`（正常验证图）同时用于**选最优权重**、**选融合权重**和**校准阈值**。`images/val` 通常全是缺陷图，没有 `normalval` 就算不出图级 AUROC，也校准不了阈值。
> 建议再单独留一份从不参与上述过程的测试图做最终验证。

训练前建议先体检一次（只报告，不改数据）：

```bash
python scripts/check_images.py --data-yaml dataset/fabric19.yaml    # 坏图/截断图
python scripts/check_leakage.py --data-yaml dataset/fabric19.yaml --show 10   # train/val 近重复
```

> 数据集目录名叫 `source5_augmented`：如果是**先离线增强、后划分 train/val**，同一张原图的翻转/旋转/调亮副本会同时落在 train 和 val，
> val 指标虚高、阈值偏松、上线后表现变差。`check_leakage` 报告近重复比例 > 10% 时，请按**原图**分组重新划分 train/val。
> （布匹纹理均匀，dHash 区分度有限，结果当线索，需要人工抽看。）

---

## 3. 四条命令

```
 ① python scripts/train_detector.py      Dinomaly 联合训练 → 自动 kNN 库 + 融合 (val 选权重) ─► checkpoints/detector/model_final.pth
 ② python scripts/calibrate.py           在 val 上扫参 + 校准正/异常阈值 ──────────────────────► checkpoints/detector/thresholds.json
 ③ python scripts/train_classifier.py    按 YOLO 缺陷框裁剪, 训练 19 类分类 ─────────────────► checkpoints/classifier/best.pt
 ④ python scripts/infer.py --input ...   原始图 → 正/异常检测 → 区域分类 ─────────────────────► runs/infer/{<图名>_vis.png, results.json}
   └ 加 --no-classify = 纯正异常检测 (不分类, 不需要 ③)
```

① → ② 必须按顺序（② 用 ① 的权重）；③ 与 ①② 互相独立，顺序随意（可在两台机器上同时训）；④ 需要 ①② 的产物，缺少 ③ 时只检测、不分类。
所有脚本的默认参数已是推荐配置，**直接运行即可，不用加一长串开关**；`--data-yaml` 默认就是 `dataset/fabric19.yaml`。

### ① 训练检测器

```bash
python scripts/train_detector.py --data-yaml dataset/fabric19.yaml --out checkpoints/detector
```

| 项目 | 说明 |
|---|---|
| 做什么 | ① Dinomaly2 联合训练：正常图走重建损失，带框缺陷图走图级/框级 margin 损失；每 2500 iter 在 val 上评估并保存最优权重。② **训练结束自动**建 kNN 记忆库（DINOv2 中层 patch + coreset，几分钟）并与 Dinomaly 融合，权重 `w` 在 val 上按 (I-AUROC, 间隙) 选，**`w=1`（纯 Dinomaly）和 `w=0`（纯 kNN）也是候选，所以在 val 上不会比单个检测器差**；融合没带来提升时 `model_final.pth` 就是纯 Dinomaly，推理没有额外开销 |
| 默认配方 | 448 · `dinov2reg_vit_small_14` · 10000 iters · bs16 · 关 rot90 · 局部损失 1.0（hard 20%，预热 2000）· 困难正常重加权 0.5 · 缺陷图框外重建 0.5 · StableAdamW + 余弦退火 |
| 输入 | `normaltrain`（纯正常）+ `train`（缺陷图 + YOLO 标注）；`val` + `normalval` 用于选权重 |
| 输出 | `model_final.pth`（**最终检测器**，②④ 默认用它，可能是纯 Dinomaly / 融合 / 纯 kNN）· `model_best.pth`（Dinomaly 最优）· `model_last.pth`（最新，可续训）· `knn_best.pth` · `model_fused.pth`（中间产物） |
| 耗时 | GTX 1660 bs16 约 2 s/iter，10000 iter 约 5.5 h；kNN + 融合几分钟（均未在 1660 实测，以日志为准） |
| 怎么看成功 | 训练日志 `I-AUROC` 上升、`gap_rel` 为正；末尾 `w_dinomaly=… I-AUROC=… 相对间隙=…` 表和 `最终检测器: …` 行告诉你选了谁 |

常用选项（都可不加）：

| 想做的事 | 加什么 |
|---|---|
| 断点续训 | `--resume checkpoints/detector/model_last.pth` |
| 训得更久 / 提前停 | `--total-iters 20000` · `--early-stop 2`（连续 2 次评估无提升就停） |
| 更准（更慢） | `--backbone dinov2reg_vit_base_14` |
| 更快 | `--enc-fp16`（冻结 encoder 用 fp16，需 CUDA；先跑几百 iter 对比 loss） |
| 不要 kNN/融合（推理最快） | `--no-knn` |
| 调 kNN | `--knn-layer-sets "5,7,9;3,5,7"`（层须在 2~9，val 上择优）· `--knn-bank-size 20000`（越大越准、推理越慢）· `--knn-max-images 300` |
| 布匹方向不固定 | `--rot90`（默认关：方向固定时关掉，细长经/纬线缺陷更敏感） |
| 训练完顺带校准（省掉 ②） | `--auto-calibrate` |
| 指定数据 / 显卡 | `--data-yaml xxx.yaml` · `--root <数据根目录>` · `--device cuda:0` |

训练配方要点：官方 Dinomaly2 的 StableAdamW（抑制 loss 尖峰、坏梯度整步跳过）+ 余弦退火；缺陷图占比过低时自动抬到 25%，保证 margin/框损失每步都有正样本；
图级 hinge 用与推理同款的分数（上采样 + 高斯平滑后取最大）。完整参数见 `python scripts/train_detector.py -h`。

### ② 校准阈值

```bash
python scripts/calibrate.py --data-yaml dataset/fabric19.yaml
```

| 项目 | 说明 |
|---|---|
| 做什么 | 在 `normalval`（正常）与 `val`（缺陷）上各前向一次，**默认已开 `--sweep`**：扫 `smooth_sigma ∈ {4,6,8,12}` × `score_ratio ∈ {最大值, 0.1%, 0.5%, 1%}`，采用正常/缺陷间隙最大的组合；`img_thr` 取 `max_normal` 与 `min_defect` 的中点；`pix_thr` 取正常图像素分数 99.99% 分位数（区域分割用） |
| 输入 | 默认 `checkpoints/detector/model_final.pth`（没有则 `model_best.pth`）；换权重用 `--det-ckpt` |
| 输出 | 权重同目录的 `thresholds.json`（`img_thr / pix_thr / smooth_sigma / score_ratio / 训练分辨率`，推理自动沿用） |
| 怎么看成功 | 日志末尾 `阈值已保存 ... separable=True`：所有正常图分数 < `img_thr` < 所有缺陷图分数；`separable=False` 见[排障](#正常缺陷分数重叠separablefalse) |
| 常用选项 | `--no-sweep` 不扫参 · `--score-ratio 0.01` 手工指定图级分数 · `--max-images 200` 限制每侧图数加快 · `--half` |

> 重训检测器、换数据或换分辨率后，必须重新运行本步。`thresholds.json` 记录训练分辨率，推理时自动核对，不一致会告警。

### ③ 训练分类器

```bash
python scripts/train_classifier.py --data-yaml dataset/fabric19.yaml --out checkpoints/classifier
```

| 项目 | 说明 |
|---|---|
| 做什么 | ① 缺陷框 + 整图 + 区域内/外特征 + 几何 → 冻结 DINOv2 抽特征（稀有类特征级过采样）；② 正常图随机框作「正常」类，让二次判断学会推翻误报；③ 余弦头 + Balanced Softmax 学「19 类 + 正常」；④ **留一类校准**：轮流把一个已知类当作「没见过的类」，学出新颖度打分（Mahalanobis++ / 原型距离 / 最大概率 / 间隔 / 能量 → 逻辑回归）和阈值 |
| 输入 | `train` / `val` 的缺陷图与 YOLO 标注、`normaltrain` / `normalval` 正常图（没有 val 时自动从 train 划 10%） |
| 输出 | `checkpoints/classifier/best.pt`（分类头 + 新颖度模型 + `nov_thr` / `normal_thr` 等阈值） |
| 耗时 | 几分钟（特征只抽一次；留一类校准每类 40 轮小头，十几类约一分钟量级，未实测） |
| 怎么看成功 | 日志 `val_bal_acc`、`推翻阈值`（val 缺陷框被推翻数应为 0）、`留一类校准 AUROC`（越高越能识别没见过的类）、「已知类误判为未知」比例 |
| 常用选项 | `--no-loco` 跳过留一类校准（未知退回最大概率阈值）· `--max-normal` 正常样本上限 · `--kind embed` 只看区域裁剪（A/B 对照）· `--kind cnn` 旧方案 |

「严重度」门槛由 ② `calibrate.py` 写进 `thresholds.json`（`severe_score` / `severe_area`：val 缺陷图分数、缺陷框面积各自的 25% 分位）：**重新 calibrate 后无需重训分类器**。

验证分类器本身（GT 框直接喂分类器，得总准确率、每类准确率、混淆矩阵、混淆对、macro 平均）：
`python scripts/evaluate.py --mode gt`

### ④ 预测（完整预测 / 纯正异常检测）

```bash
# 完整预测: 正/异常检测 + 区域分类
python scripts/infer.py --input <图片 / 图片目录 / 视频文件 / 摄像头编号> --out runs/infer
# 纯正异常检测: 只判正常/异常并框出区域, 不分类 (不需要 ③ 的分类器)
python scripts/infer.py --input <图片 / 图片目录 / 视频文件 / 摄像头编号> --no-classify --out runs/det
```

- `--input` 自动识别：文件夹 · 图片文件 · 视频（mp4/avi/mov/mkv）· 数字（摄像头编号，配合 `--show` 实时显示）。也可分别用 `--image / --dir / --video / --camera`。
- 默认读取 `checkpoints/detector/model_final.pth`（没有则 `model_best.pth`）、同目录的 `thresholds.json`、`checkpoints/classifier/best.pt`；路径不同时用 `--det-ckpt / --thresholds / --cls-ckpt`。
- 检测器/阈值缺失会直接提示该先跑哪条命令；分类器缺失则只检测不分类。
- 常用：`--half`（FP16，提速取决于显卡，1660 无 Tensor Core，请用 `benchmark.py` 实测）· `--top-k 3`（仅旧分类器 `--kind embed/cnn` 用：只分类最异常的 3 个区域；二次判断固定看全部区域）· `--no-heatmap` · `--heat-region-only` · `--heat-floor-ratio 1.3`（热力图噪点多时调大）。

---

## 4. 两种预测方式：完整预测 / 纯正异常检测

输入都是**原始整图**（图片 / 目录 / 视频 / 摄像头）：

| | 完整预测（默认，命令 ④） | 纯正异常检测 |
|---|---|---|
| 做什么 | 先判正常/异常 → 异常图框出区域 → 二次判断（整图+区域综合：19 类 / 未知类型 / 推翻为正常） | 只判正常/异常并框出异常区域，不分类 |
| 需要的权重 | 检测器（①②）+ 分类器（③） | 只要检测器（①②），不需要 ③ |
| 预测 | `python scripts/infer.py --input <图片/目录/视频> --out runs/infer` | `python scripts/infer.py --input <图片/目录/视频> --no-classify` |
| 输出 | 每图 `<名>_vis.png`（热力图 + 框 + 类名/置信度）+ `results.json` | 同上，区域类别统一标「未知类型」 |
| 评估 | `python scripts/evaluate.py --mode pipeline --top-k 0` | `python scripts/evaluate.py --mode pipeline --no-classify` |
| 评估输出 | 图级准确率 + 检测框召回 + 匹配区域分类准确率 + 混淆对 | 图级准确率、缺陷图漏检数、正常图误报数、召回/精确率、漏检/误报逐张列表 |

评估同时评 `val` 缺陷图和 `normalval` 正常图（正常图用来统计误报）。期望图级准确率 = 1.0、漏检/误报列表为空；
不达标看日志「缺陷图 score 分位数 vs img_thr」与「漏检图 margin」，重叠则回到 ①②。

## 5. 输出格式

`results.json`（每张图一项）：

```json
{"image": "001.png", "is_anomaly": true, "anomaly_score": 0.83, "anomaly_conf": 0.67, "img_thr": 0.41,
 "latency_ms": 25.1,
 "verdict": "断纬", "final_class_id": 1, "final_conf": 0.91, "judge_override": false,
 "regions": [{"box": [x1,y1,x2,y2], "area": 900, "anomaly_score": 0.9, "anomaly_conf": 0.69,
              "class_id": 1, "class_name": "断纬", "class_conf": 0.91, "classified": true, "unknown": false, "novelty": 0.03,
              "severe": true, "probs": [...]}]}
```

- `anomaly_conf = score / (score + img_thr)`，恰在阈值上为 0.5；
- `verdict` / `final_class_id` 是**整图最终结论**：`正常`（`-1`）/ 缺陷类名 / `未知类型`（`19`）；取最异常区域的结论；
- `judge_override=true`：Dinomaly2 判异常，但二次判断认为所有区域都像正常纹理，推翻为正常（`regions` 为空）；
- 区域的 `class_id == 19`（`unknown=true`）：**新颖（`novelty ≥ nov_thr`）且严重（`severe=true`）**——没见过的异常；
- 二次判断会看**全部**异常区域（上限 8 个，不受 `--top-k` 限制）；`--kind embed/cnn` 的旧分类器才用 `top_k`，其余区域标 `classified=false` 的「未知类型」；
- 热力图用**绝对阈值**渲染：低于正常图噪声上限的像素完全透明，高于它的部分颜色与透明度随强度增大，背景不会被拉伸成满屏噪点。

## 6. 验证与辅助命令

| 目的 | 命令 |
|---|---|
| 完整预测评估 | `python scripts/evaluate.py --mode pipeline --top-k 0` |
| 纯检测评估 | `python scripts/evaluate.py --mode pipeline --no-classify` |
| 分类器验证（GT 框直接喂分类器） | `python scripts/evaluate.py --mode gt` |
| 体检数据（坏图、train/val 近重复） | `python scripts/check_images.py` · `python scripts/check_leakage.py --show 10` |
| 速度测试（检测器/分类器/整体 ms 与 FPS） | `python scripts/benchmark.py --device cuda:0 --half` |
| 单独重建 kNN 记忆库 / 重做融合（日常不用，① 已自动做） | `python scripts/fit_knn.py` → `python scripts/fuse_detectors.py` → 重新 `calibrate.py --det-ckpt checkpoints/detector/model_fused.pth` |

所有脚本的 `--data-yaml` 默认都是 `dataset/fabric19.yaml`，`--root` 可覆盖数据根目录。

---

## 7. 原理简述

**检测器（Dinomaly2）**：冻结的 DINOv2 encoder 抽多层特征，bottleneck + 线性注意力 decoder 学着把它重建回来。训练时：
- 正常图：loose cosine 重建损失（只压最难的一部分 token，避免过拟合）+ 局部逐位置 cosine（细线类信号不被全局池化稀释，hard mining 取最难 20% 位置）；
- 缺陷图：图级 margin（缺陷图分数须比同 batch 正常图高 `margin`）+ 框级 margin（框内 token 距离 ≥ `margin`，直接抬高最弱缺陷的分数）；框外正常纹理也参与重建；
- 分数高的正常图重建损失加权（困难正常重加权），压低 `max_normal`。

推理：异常图 = 1 − cos(encoder 特征, decoder 重建) → 上采样到输入尺寸 → 高斯平滑；图级分数 = 异常图最大值（或最高 N% 像素均值，由 `calibrate` 选）；
异常图高于 `mask_thr` 的连通域 → 缺陷区域框 → 交给分类器（先检测后分类）。

**kNN 记忆库与融合**：正常图的 DINOv2 中层 patch 特征经 coreset 压缩成记忆库，测试 patch 到最近正常 patch 的余弦距离就是异常分数（免训练，思路来自 AnomalyDINO / PatchCore）。
重建式（对"能被重建出来的缺陷"会漏）和记忆库式（对"记忆库没覆盖的正常变化"会误报）误差模式不同，两张异常图按"正常图典型分数"归一化后加权：
`fused = w · dino_map / ref_d + (1 − w) · knn_map / ref_k`。两者共用同一个冻结 encoder，kNN 特征直接取 Dinomaly 前向里已算好的目标层，只多一次最近邻搜索。

**二次判断（`src/judge.py`）**：Dinomaly2 的「异常」只是线索，不是结论。对每个异常区域，用冻结 DINOv2 抽 7 段 L2 归一化特征再拼几何：
区域裁剪的 `[CLS, 末层 patch 均值, 中层 patch 均值]`、**整图** CLS、整图 token 里**区域内**的均值（重点）与**区域外**的均值（其余区域也参与）、区域面积与宽高比。
余弦头（Balanced Softmax，稀有类特征级过采样）在这个拼接特征上学出 `19 类 + 正常`，所以它能同时回答「是不是异常」与「是哪一类」：
- 全部区域 `p(正常) ≥ normal_thr` → 推翻 Dinomaly2。`normal_thr` 取 val 缺陷框 `p(正常)` 的最大值之上，宁可少推翻也不把真缺陷判成正常；
- 否则取最像的缺陷类；若同时 **新颖** 且 **严重** → 「未知类型」。

**未知异常 = 新颖度 AND 严重度**：
- 严重度来自 Dinomaly2：图级分数 ≥ `severe_score` 且区域面积 ≥ `severe_area`（val 缺陷里最弱 25% 以下的不算「较大且严重」）。很小/很弱的区域即使不像任何类，也给最接近的已知类；
- 新颖度来自 5 个开集分数：最近类 Mahalanobis 距离（L2 归一化特征 + PCA + 共享协方差，即 Mahalanobis++ 式）、到最近类原型的余弦距离、最大概率、top1−top2 间隔、能量，由逻辑回归融合。训练数据来自**留一类**：轮流把一个已知类从分类头与类统计里拿掉，它的 val 框当「没见过的类」正例、其余已知类当负例，所以**不需要真实的未知样本**；`nov_thr` 取已知类 95% 分位。
依据：OpenOOD v1.5 基准中 kNN / Mahalanobis / ViM 家族是后验开集检测最强的一批，Mahalanobis++（2025）证明特征 L2 归一化可让 Mahalanobis 在 ViT 上稳定提升；用伪未知类校准开集阈值是开放集识别的常规做法。

> 局限：留一类模拟的是「训练里缺一类」，真实未知缺陷可能和已知类差别更小或更大；**未在真实布匹数据上实测**，请看日志的留一类 AUROC 与 `evaluate.py --mode gt` 的结果。

旧的单区域分类器（`--kind embed`）仍保留作 A/B：只看区域裁剪，最大概率 < `unknown_thr` 判「未知类型」。

## 8. 调参与排障

### 正常/缺陷分数重叠（`separable=false`）

先看 calibrate 日志的「各缺陷类最低图分数」定位是哪些类（常见是细长经线类），然后依次：

1. 加大 `--total-iters`、`--local-loss-w 2`（局部损失 / 困难正常 / 框外重建已是默认）；
2. 确认没有开 `--rot90`（方向固定的布匹应保持默认关）；
3. 先重新 `calibrate.py`（自动在 `smooth_sigma × score_ratio` 里挑间隙最大的组合）；仍重叠再考虑训练时 `--smooth-sigma 8~12`；
4. 确认 `normaltrain` 覆盖全部正常变化（暗角/光照/品种）；
5. 仍只差一点：`--image-size 560`（14×40，合法尺寸），需重训 + 重校准。

### 按症状选下一步

| 症状 | 手段 | 代价 |
|---|---|---|
| 检测**漏报**（`min_defect` 低） | 细长缺陷保持局部损失；更大 backbone `dinov2reg_vit_base_14`；更高分辨率（`--image-size 560/672`，公开结果里边际收益小） | 后两者在 1660 上明显变慢 |
| 检测**误报**（`max_normal` 高） | 跑 `check_leakage.py`（val 虚高会让阈值偏松，上线反而误报）；补更多、更多样的正常图；`--hard-normal-w` 加大 | 基本无推理代价 |
| **分类**准确率低 | 看 `evaluate --mode gt` 的混淆对；分类器换 `--backbone dinov2reg_vit_base_14`；`--kind cnn` 做 A/B；补稀有类数据 | 更大骨干每个裁剪约 3× 计算 |
| 端到端低但 gt 分类高 | 检测框质量问题：用 `evaluate --mode pipeline` 看框召回；调 `--region-score-ratio` / `box_pad` | 无 |
| 推理太慢 | `--half`；训练时 `--no-knn` 去掉 kNN 融合（再 `calibrate`）；`--image-size 336` | 精度可能下降 |

### 热力图噪点多

- 显示层：`--heat-floor-ratio 1.3~1.6`，或 `--heat-region-only`；
- 分数层：`--smooth-sigma 8~12` 重训/重校准；
- 区域框层：`min_area_ratio`（默认 2e-4）调大可去掉零星小框。

### 其它

| 现象 | 处理 |
|---|---|
| 训练/校准/推理分辨率不一致 | `thresholds.json` 记录分辨率，`infer` 加载时自动核对并告警；默认 448 勿改 |
| Windows 训练卡住 | 已默认 `--num-workers 0`，可手动调大 |
| 中文路径/文件名 | 全部走 Unicode 安全读写，无碍 |
| 校准太慢 | `--max-images 200` 先抽样 |
| backbone 权重下载失败 | 手动下载放到 `backbones/weights/`，或 `--weights-dir` 指定 |
| kNN/融合那步报错 | 训练会自动退回纯 Dinomaly（`model_final.pth` = `model_best.pth`）；之后可单独跑 `fit_knn.py` + `fuse_detectors.py` |
| 想恢复旧默认训练参数 | `--local-loss-w 0 --hard-pos-ratio 0 --hard-normal-w 0 --local-loss-warmup 0 --defect-bg-w 0 --rot90 --total-iters 20000 --eval-interval 5000 --no-knn` |

> ⚠️ kNN/融合的收益来自 AnomalyDINO 等公开结果与"融合在 val 上不会比单个差"的构造，**在你的布匹数据上的实际提升没有实测**；
> 融合会让推理多一次最近邻搜索，请用 `benchmark.py --half` 在 1660 上看延迟。嫌慢就 `--no-knn` 重训，
> 或 `cp checkpoints/detector/model_best.pth checkpoints/detector/model_final.pth` 后重新 `calibrate.py`。
> 选融合权重、选层组和校准阈值用的是同一份 val，val 图少时间隙会偏乐观，务必用独立测试图做最终验证。

## 9. 项目结构

```
scripts/
  train_detector.py     ① 检测器训练 (+ 训练末尾自动 kNN 建库与融合, 写 model_final.pth)
  calibrate.py          ② 阈值校准 (默认扫参)
  train_classifier.py   ③ 分类器训练
  infer.py              ④ 预测: 图片 / 目录 / 视频 / 摄像头; 完整预测或 --no-classify 纯检测
  evaluate.py           评估: --mode pipeline (完整 / --no-classify 纯检测) · --mode gt (分类器)
  benchmark.py          速度测试
  check_images.py       坏图体检
  check_leakage.py      train/val 近重复(数据泄漏)检查
  fit_knn.py            单独重建 kNN 记忆库 (日常不用)
  fuse_detectors.py     单独重做 Dinomaly+kNN 融合 (日常不用)
src/
  detect.py             检测器: 训练 / 评估 / 校准 / 推理 / 区域分割
  knn_detect.py         kNN 记忆库检测器 (DINOv2 中层 patch + coreset)
  fusion.py             Dinomaly + kNN 融合, 训练后自动融合流程
  classify.py           分类器公共部分: 裁剪数据集 / 未知类型校准 / TTA 推理 / CNN 方案
  embed_classify.py     DINOv2 嵌入分类器 (只看区域裁剪, 余弦头 + Balanced Softmax; 二次判断复用其分类头训练)
  judge.py              二次判断: 整图+区域+几何特征 / 推翻误报 / 留一类校准的未知异常判定
  pipeline.py           端到端流水线: 检测 → 区域 → 二次判断 (或旧式 top_k 分类) → 热力图可视化
  dinomaly/             Dinomaly2 模型 / 损失 / 优化器 / DINOv2 encoder
  data/                 YOLO 数据集读取 (Unicode 安全, 坏图跳过) 与配置解析
  util.py               日志 / 指标 / 可分离高斯平滑 / 框与掩码工具 / 中文字体
dataset/fabric19.yaml   数据集路径与 19 类类名 (改成你的实际路径)
tests/                  smoke_test.py (全链路冒烟) + test_detector_fixes.py (回归测试) + test_judge.py (二次判断)
```

## 10. 测试

```bash
python tests/smoke_test.py            # CPU 可跑, tiny_test_14 随机初始化: 训练/校准/推理/分类/可视化全链路
python tests/test_judge.py             # 二次判断: 训练(含留一类校准)/存取/推翻误报/未知异常/严重度门槛
python tests/test_detector_fixes.py   # 回归测试 (秒级): 增强与掩码同步 / 平滑核 / 校准 / StableAdamW / 调度器 / 扫参 /
                                      #   kNN / 融合 / 分类头 / 训练后自动融合 / 命令行参数存在
```

沙箱里只有 CPU 和合成数据，三套测试只验证流程与数值等价，**不代表真实数据上的精度或速度**。
