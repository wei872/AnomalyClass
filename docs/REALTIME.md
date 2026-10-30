# 实时推理指南（GTX 1660 · 40 FPS）

规划目标：1600×846 原图整幅输入（网络输入 832×1600），端到端 ≥ 40 FPS。
本文硬件数字仅为历史示例或理论估算，**不是已验证的性能或精度保证**；需在目标产线上实测。

## 1. 单帧预算怎么切

40 FPS 时一帧只有 25 ms。四段开销必须并行（三工位流水线），而不是串起来：

| 阶段 | 内容 | 1660 上预期 | 说明 |
|---|---|---|---|
| 读图 | JPEG 解码 1600×846 | 6–12 ms | PNG 会到 30–60 ms，产线请用 JPEG/BMP 或直接拿相机内存 |
| 前向 | 网络 + 记忆库 | 5–7 ms | 832×1600, FP16, NHWC（3.14 GMACs） |
| 后处理 | 区域分析 / 连通域 / 类别池化 | 2–4 ms | 分析图 400×208（832×1600 输入时） |
| 可视化 | 热力图叠加 + PNG 保存 | 8–20 ms | **不要放在主循环里**，异步或关闭 |

流水线（`fabric_ncd/realtime.py`）把三段放到三个线程，队列容量 2：

```
读图线程 ──▶ 前向线程(GPU) ──▶ 后处理线程(CPU) ──▶ results()
```

理想情况下吞吐受最慢阶段限制，端到端延迟还包括各阶段排队、调度和同步；实际吞吐与延迟应以压测为准。

## 2. 怎么测（在你自己的机器上）

```bash
# 单帧各段耗时 + 流水线吞吐（含达标判定与优化建议）
python tools/rt_bench.py --ckpt runs/fabric19_fabricncd_s/best.pt \
    --source /data/fabric/line --frames 300 --fp16

# 只看单帧串行延迟
python tools/rt_bench.py --ckpt best.pt --source a.jpg --frames 100 --mode sequential

# 结果落盘，便于对比不同参数
python tools/rt_bench.py --ckpt best.pt --source /data/line --frames 300 --json runs/rt.json
```

输出示例（本机 2 核 CPU，仅示意格式）：

```
阶段             mean      p50      p95   (ms)
read          115.83   115.83   150.02
infer          99.23    99.23   111.79
analyze         5.33     5.33     5.71
吞吐: 8.7 FPS  (目标 40)  ->  未达标 ✘
```

看到 `未达标` 时，按提示逐条降配（见第 4 节）。

## 3. 已经做了哪些优化（默认全开）

| 优化 | 位置 | 收益 |
|---|---|---|
| 输出分数图 1/4 下采样（`infer.out_stride`） | `inference.py` | 原本每帧要上采样 19 通道类别图 + 64 通道嵌入，现在只算 400×208，前向从 906 ms 降到 58 ms（该组为 1024×1728 输入下的 CPU 实测，趋势不变） |
| 类别概率再降一档（`infer.probs_mul`） | `ncd.py` | 类别图只做区域平均，不需要和分数图同分辨率，省掉 19 通道全分辨率的双线性上采样 |
| 记忆库查询池化（`infer.memory_stride`） | `memory.py` | 查询点数 ÷4（2048×64 的点积量 ÷4），分数图上采样回去，区域检出基本无损 |
| 织造周期缓存（`infer.weave_interval`） | `inference.py` | FFT 从每帧一次变成每 30 帧一次 |
| 预处理上 GPU（`infer.prep_gpu`） | `inference.py` | CPU 的 PIL 缩放 + 归一化（1600×846→832×1600 约 2–3 ms）变成 GPU 0.3 ms |
| FP16 + NHWC（`infer.fp16` / `channels_last`） | `inference.py` | 1660 无 Tensor Core，但 FP16 双发射仍有 ~1.5×；NHWC 让 cudnn 走更快的深度可分离卷积内核 |
| 形态学改可分离平移 + 游程连通域 | `ops.py` | 610×360 上形态学 5.1→0.2 ms，连通域 7.5→1.5 ms（CPU 实测；400×208 上再降 2.2 倍） |
| 三线程流水线 | `realtime.py` | 读图 / 前向 / 后处理重叠，吞吐由"三者之和"变成"最慢一段" |

## 4. 降配顺序（不够快时依次尝试）

1. `data.short_side`: 1024 → 896 → 768（`image_size: auto` 时只改这一个数, 长宽比与宽高顺序自动保持; 每档算力 ×0.75 / ×0.56）；
2. `infer.out_stride`: 4 → 6（分析图 407×240，后处理再降 2.2 倍）；
3. `infer.memory_stride`: 2 → 4（记忆库再降 4 倍，对小疵点略有影响）；
4. `model.backbone.arch`: `mobilenet_v3_small` → `lite`（纯 PyTorch 主干，无预训练，精度略低）；
5. 关掉 `texture.use_imaginary`（Gabor 核数量减半）。

不要用"降 batch"来提吞吐：实时场景 batch 恒为 1，减 batch 没有意义。
也不要靠切块来提速——切块是**提精度**用的，一帧切成 9 块只会更慢。

## 5. 精度代价

- `out_stride=4`：定位框会受地图分辨率量化影响，细小区域的召回需实测；
- `memory_stride=2`：小面积（< 8×8 px）疵点的异常分数会略微平滑，建议 ≥ 8 px 的疵点用默认值；
- `weave_interval=30`：布面纹理在 1 s 内突变（换布种）时，最多滞后 30 帧重新估计；
- `fp16`：数值偏差取决于硬件和输入，阈值附近的结果可能变化；应在目标设备上验证。

切块评估使用 `infer.out_stride` / `infer.probs_mul` 和相同的区域分析；训练端校准仍采用训练输入。
改变尺寸、切块或 stride 时需要重新验证阈值。`eval.analysis_stride` 应与 `infer.out_stride` 对齐。

## 6. 常驻服务写法

```python
from fabric_ncd.inference import FabricPredictor
from fabric_ncd.realtime import RealtimePipeline

predictor = FabricPredictor("runs/fabric19_fabricncd_s/best.pt", device="cuda", fp16=True)
with RealtimePipeline(predictor, queue_size=2) as pipe:
    for frame in camera:      # HWC ndarray、CHW tensor、路径或 PIL 图
        pipe.submit(frame)
        while (res := pipe.poll()) is not None:
            if res.error:
                print("处理失败:", res.error)
            elif res.regions:
                print(res.index, res.latency_ms, res.regions)
```

非阻塞 `submit` 只在输入队列满时丢弃最旧的待处理帧，可通过 `pipe.dropped` 统计；
正在处理的旧帧仍会完成。`block=True` 对所有阶段施加背压、不主动丢帧。失败帧以带 `error`
的结果返回，可能比更早的成功帧先返回。输出队列无界，务必持续消费。`stop()` 是立即停止，
不会保证处理完在途帧，也不能重启同一实例；有限批次需先消费全部预期结果再停止。
