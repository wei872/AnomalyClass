# AnomalyClass — 布匹缺陷检测 + 分类流水线

基于 Dinomaly2 的异常检测器：**正常样本和带框缺陷样本联合训练**（同一 batch 混合：正常图走重建损失，缺陷图走 margin/box 监督损失，不是分开训练两个模型），固定 **448** 输入且 **不裁边缘** crop_size=image_size=448——四周暗角属于正常外观，边缘疵点也不能丢；GTX 1660 实时。
推理顺序固定为**先检测后分类**：检测器圈出异常区域 → 再用轻量 CNN 分类每个区域（19 类 + 「未知类型」）。YOLO 标注的疵点框既是联合训练里缺陷侧的监督信号、分类器的训练样本，也是验证标准。

## 目录结构

```
src/detect.py        Dinomaly2 检测器: 训练 / 校准(间隙中点) / 推理 / 分割
src/classify.py      缺陷分类器: YOLO框裁剪训练 (letterbox保宽高比), 未知类型校准, TTA推理
src/pipeline.py      端到端流水线: 检测 -> 区域 -> top_k 分类 -> 可视化
src/data/            YOLO 数据集读取 (Unicode安全, 坏图跳过), 配置解析
src/dinomaly/        Dinomaly2 模型与损失 (含 local_cosine 局部损失)
scripts/             train_detector / calibrate / infer / train_classifier / evaluate / benchmark / check_images
configs/fabric19.yaml  推荐设置参考 (脚本默认值已与之一致)
dataset/fabric19.yaml  数据集路径与 19 类类名 (改成你的实际路径)
tests/smoke_test.py  冒烟测试 (tiny 随机初始化, CPU 可跑)
```

## 安装

```bash
pip install torch torchvision numpy opencv-python-headless pyyaml pillow scikit-learn scipy tqdm
# DINOv2 backbone 预训练权重放到 backbones/weights/ (如 dinov2_vits14_reg4_pretrain.pth)
# 下载地址: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
```

## 数据准备

编辑 `dataset/fabric19.yaml`（Windows 路径示例见文件注释）:

```
DATASET_ROOT/
  images/train/  images/val/     # 带疵点的图
  labels/train/  labels/val/     # YOLO txt: cls cx cy w h (归一化)
  normal/images/train/           # 纯正常图 (与 images/train 缺陷图一起参与检测器联合训练)
  normal/images/val/             # 纯正常图 (阈值校准用)
```

## 使用流程（按顺序）

### 1) 联合训练正异常检测器（normal 正常图 + train 缺陷图一起喂，同一 batch 混合）

```bash
python scripts/train_detector.py --data-yaml dataset/fabric19.yaml --out checkpoints/detector
# 默认 448 / dinov2reg_vit_small_14 / 20000 iters / bs16 (1660 约 11h)
# 损失 = 正常图重建 (loose cosine) + 缺陷图监督 margin (图级 hinge --margin-w/--margin + 框级 hinge --box-margin-w), 默认已开
# 断点续训: --resume checkpoints/detector/model_last.pth
# 细长经线类缺陷响应弱时加: --local-loss-w 1.0 --hard-pos-ratio 0.2 ; 压 max_normal 可加 --hard-normal-w 0.5
# 消融对照才用 --no-defect-sup (纯无监督), 正式训练保持联合
```

### 2) 校准阈值（关键：正常/异常各自 100%，不靠调参跷跷板）

```bash
python scripts/calibrate.py --data-yaml dataset/fabric19.yaml \
    --det-ckpt checkpoints/detector/model_best.pth --out checkpoints/detector/thresholds.json
```

- 在 `normal/images/val` 上得正常分数 `max_normal`，在 `images/val`（带标注缺陷图）上得缺陷最低分 `min_defect`；
- 若 `min_defect > max_normal`（分布可分）→ `img_thr = (max_normal + min_defect)/2` **间隙中点**，两侧最大 margin，**所有正常图判正常、所有缺陷图判异常**；
- 若重叠 → 日志告警 `separable=false`，取 `max_normal` 并提示重训检测器（更多 iters / `--local-loss-w 1.0`)，让分布分开；
- `pix_thr` 取正常图异常像素 99.99% 分位数，用于区域分割。

### 3) 先跑正异常检测（不分类），看结果图

```bash
# 纯检测 (不加载分类器, 更快): 在 val 缺陷图 + normalval 上分别跑
python scripts/infer.py --data-yaml dataset/fabric19.yaml --no-classify \
    --det-ckpt checkpoints/detector/model_best.pth --thresholds checkpoints/detector/thresholds.json \
    --dir D:/.../images/val --out runs/det_val
python scripts/infer.py --data-yaml dataset/fabric19.yaml --no-classify \
    --det-ckpt checkpoints/detector/model_best.pth --thresholds checkpoints/detector/thresholds.json \
    --dir D:/.../normal/images/val --out runs/det_normal
# 每图输出 <名>_vis.png (热力图+框), results.json 含 is_anomaly/anomaly_score/img_thr/regions
# 热力图只显示 > pix_thr*heat-floor-ratio 的像素 (绝对阈值, 不再按每图分位数拉伸); 噪声还多: --heat-floor-ratio 1.5, 或 --heat-region-only 只在框内显示
```

验证正异常二分类（图级准确率 + 漏检/误报逐张列表）:

```bash
python scripts/evaluate.py --data-yaml dataset/fabric19.yaml --mode pipeline \
    --det-ckpt checkpoints/detector/model_best.pth --thresholds checkpoints/detector/thresholds.json --top-k 0
```

期望：图级准确率 = 1.0，漏检/误报列表为空。若不达标看日志里「缺陷图 score 分位数 vs img_thr」与「漏检图 margin」——重叠则回到第 1/2 步重训+重校准。

### 4) 检测之后再训练分类器（在 YOLO 疵点框裁剪上）

```bash
python scripts/train_classifier.py --data-yaml dataset/fabric19.yaml --out checkpoints/classifier
# 默认最轻量 mobilenet_v3_small @224, 50 epochs (早停); 预训练权重自动下载, 国内慢/损坏时用
#   --pretrained-path 指向浏览器手动下载的本地 .pth
# 区域尺寸一致性: 检测到的区域框按 box_pad=0.3 扩展 (与训练裁剪 crop_pad=0.3 相同),
# 再 letterbox 到 crop_size=224 喂分类器 —— 训练与推理同一套裁剪, --crop-size 不用调。
# letterbox 保宽高比 + 类别加权 + MixUp + 早停; 低于 unknown_thr 自动判「未知类型」
```

验证分类器本身（GT 疵点框直接喂分类器 → 准确率/混淆矩阵）:

```bash
python scripts/evaluate.py --data-yaml dataset/fabric19.yaml --mode gt --cls-ckpt checkpoints/classifier/best.pt
```

### 5) 端到端推理（先检测区域，再分类 —— 与训练后的顺序一致）

```bash
python scripts/infer.py --data-yaml dataset/fabric19.yaml \
    --det-ckpt checkpoints/detector/model_best.pth --cls-ckpt checkpoints/classifier/best.pt \
    --thresholds checkpoints/detector/thresholds.json \
    --dir D:/.../images/val --out runs/infer          # 目录; 单图 --image x.jpg; 视频 --video f.mp4; 摄像头 --camera 0 --show
# --half 开 FP16 (1660 约 2x 加速); --top-k 1 只对最异常的 1 个区域分类 (默认), 0=全部
```

## 输出格式（results.json）

```json
{"image": "001.png", "is_anomaly": true, "anomaly_score": 0.83, "anomaly_conf": 0.67, "img_thr": 0.41,
 "latency_ms": 25.1,
 "regions": [{"box": [x1,y1,x2,y2], "area": 900, "anomaly_score": 0.9, "anomaly_conf": 0.69,
              "class_id": 1, "class_name": "断纬", "class_conf": 0.91, "classified": true, "probs": [...]}]}
```

- `anomaly_conf = score/(score+img_thr)` ∈ [0,1)，恰在阈值上为 0.5；
- `class_id == 19` 为「未知类型」（分类器最大 softmax < 自动校准的 unknown_thr，或该区域未送分类器 `classified=false`);
- 只对最异常的 top_k 个区域做分类，其余区域照常输出但标未知类型。

## 性能（GTX 1660）

```bash
python scripts/benchmark.py --data-yaml dataset/fabric19.yaml --device cuda:0
```

448 + ViT-S 约 30~60 FPS（`--half` 再翻倍）；检测器训练 bs=16 约 2s/iter。

## 常见问题

- **热力图满屏噪点**: ① 显示层已改为绝对阈值 (低于正常图噪声上限 `pix_thr` 的像素不上色), 调 `--heat-floor-ratio` / `--heat-region-only`; ② 想从分数本身压噪点, 用 `--smooth-sigma 8~12` 重训/重校准 (旧版 5×5 核下 sigma 4 和 8 几乎无差别, 已修复).

- **Windows 训练卡住**: 已默认 `--num-workers 0`（主进程加载，无 spawn 子进程重 import 问题），可手动调大。
- **中文路径/中文文件名**: 全部走 Unicode 安全读写（imread_unicode/imwrite_unicode），无碍。
- **坏图/截断图**: 数据集构建时 PIL 头校验自动跳过，日志有计数。
- **检测器为何联合训练**: 正常图教模型"正常长什么样"（重建损失压低正常分数）, 缺陷图+框教模型"缺陷在哪、分数要多高"（margin/box 监督损失）, 两者同一 batch 混合训练, 直接对着 calibrate 的 `max_normal/min_defect` 缺口优化 ——
  - 图级 hinge (`--margin-w`, `--margin`): 缺陷图分数须比同 batch 正常图高出 margin;
  - 框级 hinge (`--box-margin-w`): 缺陷框内 token 平均距离须 ≥ margin, 直接抬高 `min_defect`（细线类关键）;
  - 困难正常重加权 (`--hard-normal-w 0.5`): 当前分数偏高的正常图重建损失加大, 压 `max_normal`;
  - 局部损失可预热 (`--local-loss-warmup 2000`) 避免初期不稳; `--no-defect-sup` 仅消融对照用。
- **calibrate 告警「正常/缺陷分数重叠」(无法同时 100%)**: 先看日志里「各缺陷类最低图分数」定位是哪些类重叠（常见是细长经线类），然后：
  1. 重训检测器更多 iters，细线缺陷加 `--local-loss-w 1.0 --hard-pos-ratio 0.2`（margin/box 默认已开）;
  2. 试 `--smooth-sigma 8`（训练/校准/推理三处一致；sigma>4 时平滑核宽随 sigma 增长，sigma≤4 仍是原版 5×5 以兼容旧阈值，改 sigma 后须重新 calibrate）：加大异常图平滑核，压掉正常图的单点噪声峰值，成片/成线的缺陷响应保留，可把两类分数分布拉开；
  3. 确认 normaltrain 覆盖了全部正常变化（暗角/光照/品种），整图 448 进网络不裁边缘；
  4. 若仍只差一点，可把 image_size 提到 560（14×40，仍是合法尺寸）。
- **校准太慢**: 加 `--max-images 200` 先抽样校准。
- **阈值分辨率不匹配**: calibrate 会把训练分辨率写进 thresholds.json，infer 加载时自动核对并告警 ⚠️——训练/校准/推理分辨率必须一致（默认 448，勿改）。
- **backbone 权重下载失败**: 放到 `backbones/weights/`，或训练时 `--allow-random-init` 仅冒烟测试用。
- **分类器预训练下载损坏(hash 校验失败)**: 会自动删缓存重试一次；仍失败用 `--pretrained-path` 指向手动下载的本地 .pth。

## 冒烟测试

```bash
python tests/smoke_test.py    # CPU 可跑, tiny_test_14 随机初始化, 验证训练/校准/推理/分类/可视化全链路
```
