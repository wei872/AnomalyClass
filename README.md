# FabricDefectNet — 轻量型端到端布匹缺陷检测-定位-分类-未知缺陷识别一体机

针对 `dataset/fabric19.yaml`（19 类布匹缺陷）设计的一次训练、直接工业部署的轻量模型。
**不使用 YOLO**：采用 *DINOv2-ViT-S/14 自监督骨干 + 锚自由密集检测头 + 原型度量分类头（开集）*，
单次前向同时输出 **检测框、目标置信度、19 类分类、未知缺陷判定**。

```
                       ┌──────────────────────────────────────────────┐
  布匹图 (任意分辨率)   │  滑窗 tile (默认 448, 重叠 25%, 跨 tile 合并) │
        └──────────────►                                              │
                       │  DINOv2 ViT-S/14  (21.9M, 自监督预训练)       │ stride 14
                       │     (block8 + block12 特征拼接: 细纹理+语义)  │
                       │      └─ 1x1 投影 + 2x 上采样 轻量 Neck        │ stride 7
                       │           └─ 共享卷积 trunk (GroupNorm)       │
                       │               ├─ obj  分支: 置信度 (FCOS 式)  │
                       │               ├─ box  分支: l,t,r,b → xyxy    │
                       │               └─ emb  分支: 128-d L2 嵌入     │
                       │                     └─ ×19 可学习类原型(余弦) │
                       └──────────────┬───────────────────────────────┘
                                      ▼
              每位置: 框 + conf + 19 类余弦相似度  →  NMS  →  最终结果
              max 余弦 < tau_unknown  ⇒  「未知缺陷」(没见过的缺陷也报得出来)
```

## 为什么这样设计（对照需求）

| 需求 | 方案 |
|---|---|
| 轻量 + 实时 | ViT-S/14 是 DINOv2 最轻量档；单尺度 stride-7 密集头，无 ROI/无解码器堆叠；全模型 23.0M 参数 / 69 GFLOPs@448（336 输入仅 ~39G），GPU FP16/TensorRT 下单 tile 毫秒级，滑窗并行可跑满产线相机；追求更高帧率可 `--set input_size=336` |
| 泛化强（不用 YOLO） | DINOv2 自监督特征 + 监督对比学习(SupCon) 度量嵌入 + 强增广（随机窗口裁剪/光照/噪声/离焦/小角度旋转）；锚自由设计无 anchor 先验偏差 |
| 没见过的缺陷也要分类 | 分类头是「嵌入 ↔ 类原型」余弦度量而非 softmax 全连接：已知类聚成紧簇，未见过的缺陷落在所有原型之外 → 判为 `未知缺陷` 并照常给框；阈值 `tau_unknown` 训练结束在 val 上自动标定。温度为**固定** τ=0.1（原型 InfoNCE）：可学习 logit-scale 会走「放大 scale、不对齐嵌入」的捷径，使余弦失去开集判别力（本仓库实现已规避该陷阱） |
| 精准检测+定位+分类 | 端到端单网络：GIoU+L1 回归框、focal 置信度、余弦 CE+SupCon 分类，一次前向全输出 |
| 训练一次直接可用 | `train.py` 一条命令跑完：训练→选 best→**阈值自动标定**→部署级复评→保存 `best.pt`(内嵌类别名/阈值/配置)→自动导出 `best.onnx`+`best.torchscript.pt`+`deploy_config.json` |
| 工业化使用 | ONNX(opset17, 动态 batch)/TorchScript；GroupNorm 保证 batch=1 数值稳定；中文路径/中文标签可视化/视频流/目录批量/JSON 输出/FPS 基准全内置 |

## 快速开始

```bash
pip install -r requirements.txt

# 1) 训练 (yaml 里是别人的 Windows 路径? 用 --data-root 指向你机器上的数据集根目录即可)
python train.py --data dataset/fabric19.yaml --data-root /path/to/source5_augmenteds

# 2) 推理 (权重或 ONNX 二选一, 行为一致)
python -m fabricdet.infer --ckpt runs/fabric19/best.pt          --source 图片或目录
python -m fabricdet.infer --ckpt runs/fabric19/deploy/best.onnx --source rtsp://... --fps

# 3) 复评
python eval.py --ckpt runs/fabric19/best.pt --data dataset/fabric19.yaml --data-root ...
```

数据集目录约定（与 yaml 一致）：

```
ROOT/
  images/train/  images/val/            # YOLO 图
  labels/train/  labels/val/            # cls cx cy w h (归一化)
  normal/images/train/  normal/images/val/   # 纯正常图(无标签), 用于背景抑制
```

### SDK 用法（产线集成）

```python
from fabricdet.infer import Detector
det = Detector("runs/fabric19/deploy/best.onnx")      # 或 best.pt
for p in det.predict("cam_frame.jpg"):                # BGR ndarray 亦可
    print(p["name"], p["name_en"], p["score"], p["box"], p["unknown"])
    # unknown=True 即「未知缺陷」: 没见过的新缺陷类型
det.save_vis("cam_frame.jpg", "vis.jpg")              # 中文标注可视化(自动找中文字体)
```

TensorRT 部署（可选，进一步提速）：

```bash
trtexec --onnx=runs/fabric19/deploy/best.onnx --saveEngine=best.fp16.trt --fp16
```

## 训练流程（一次命令内部自动完成）

1. **数据**：随机窗口裁剪（等价部署滑窗，解决高分辨率小缺陷）+ 光度/退化增广；正常图作纯背景负样本。
2. **分阶段训练**：前 `freeze_epochs` 冻结骨干让头部适配 → 解冻端到端微调（骨干 1/10 lr）；AdamW + 余弦调度 + warmup + EMA + AMP(GPU)。
3. **选 best**：每 `eval_every` 轮整图快速 mAP50 选 best。
4. **阈值自动标定**：val 集上用部署同款滑窗推理，扫 conf 取 F1 最优；取 TP 余弦相似度 2% 分位作 `tau_unknown`（保证 ≥98% 已知缺陷不被误判未知）。标定结果写进权重与 JSON，**部署零调参**。
5. **部署级复评 + 导出**：mAP50 / mAP75 / FPPI / per-class AP 打印并写入 `runs/fabric19/summary.json`；导出 ONNX（自动与 PyTorch 输出比对校验）+ TorchScript + `deploy_config.json`。

## 预训练权重

`backbone_ckpt: auto` 会自动下载官方 `dinov2_vits14_pretrain.pth`（~84MB）并缓存到
`~/.cache/fabricdet/`。**离线/内网机器**：提前下载后设环境变量
`FABRICDET_DINOV2_CKPT=/path/dinov2_vits14_pretrain.pth`，或 `--set backbone_ckpt=/path/...`；
完全拿不到权重时自动退化为从头训练（epochs 自动 ×1.5 补偿），流水线不中断。

## 常见问题一次性清单

| 问题 | 已内置的解法 |
|---|---|
| yaml 里是 Windows 绝对路径 | `--data-root` / 环境变量 `FABRIC19_ROOT` 自动重建路径；路径不存在时报错信息直接告诉你缺哪个目录 |
| 原图 2K/4K、缺陷只有几个像素 | 训练随机窗口裁剪 + 推理滑窗 tile(重叠 25%) + 跨 tile 合并 NMS；stride=7 保证小目标定位分辨率 |
| 类别不均衡（有的类只有几十张） | sqrt 频率加权 CE（clip 0.5~4 防失控） |
| 灰度/低对比度/相机差异 | 亮度/对比度/高斯噪声/离焦/降采样复原 增广 |
| 中文路径、中文类别名 | 全链路 `np.fromfile+imdecode` 读图、utf-8 读写；可视化自动搜索中文字体，无字体时退化为英文别名(`names_en`) |
| 标签缺失/坏图/非法类别 id | 缺失标签按正常图、坏图跳过并告警、非法 id 直接报错定位到文件 |
| 部署 batch=1 数值漂移 | 全网络 GroupNorm/LayerNorm，无 BatchNorm |
| 同一缺陷跨 tile 重复报 | 已知类优先 + 分数排序的贪心合并 NMS |
| 未知缺陷与已知框冲突 | 合并时已知类优先升级标签；未知仅在没有已知解释时才保留 |
| 阈值要人工调 | 训练结束自动标定 conf / tau_unknown 并固化进权重与 deploy_config.json |
| 无 GPU 机器 | CPU 自动关闭 AMP 正常训练/推理；ONNX CPU 后端可用 |
| 导出 ONNX 与训练行为不一致 | 导出后自动 onnxruntime 比对（偏差 >5e-3 告警）；后处理为同一份 numpy 源码 |
| 训练中断 | `best.pt/last.pt` 均落盘；`--resume` 可续（权重+meta 自包含） |

## 自检（无真实数据时验证环境/流水线）

```bash
# 一键全链路自检 (合成数据->训练->标定->导出->双后端一致性->开集规则->mAP 指标), 全过打印 ALL PASS
python tools/selftest.py

# 或手动: 合成数据 + 短训练
python tools/make_synthetic_dataset.py --out data/synthetic_fabric19 --train 60 --val 12
python train.py --data data/synthetic_fabric19/synthetic_fabric19.yaml --set epochs=20
```

## 目录结构

```
fabricdet/
  dinov2.py        DINOv2 ViT-S/14 (官方 checkpoint 兼容) + 下载/缓存
  model.py         FabricDefectNet: 骨干+Neck+密集头+原型头
  losses.py        focal/GIoU/L1/余弦CE/SupCon + FCOS assign
  data.py          数据集/增广/路径兜底/类别权重
  postprocess.py   numpy 后处理 (torch 与 ONNX 共用) + 开集规则 + 跨 tile 合并
  metrics.py       mAP50 / mAP75
  engine.py        训练循环/EMA/调度/阈值标定/部署级评估
  export.py        ONNX/TorchScript/deploy_config.json
  infer.py         Detector SDK + CLI (图/目录/视频/rtsp, 可视化, FPS)
train.py / eval.py 一次训练 / 独立复评
configs/fabric19_train.yaml 超参(默认即工业默认)
tools/make_synthetic_dataset.py 合成数据自检
```

## 已知边界

- 多卡 DDP 未内置（单卡 + 梯度累积已覆盖常见工业卡）；如需多卡可用 `torchrun` 包 `train.py` 并自行包 DDP（模型为标准 nn.Module）。
- `tau_unknown` 的语义是「与 19 类都不像」；若新缺陷与某已知类视觉极相似，会归入该已知类（这是任何开集方法的共同边界）。
