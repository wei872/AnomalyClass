"""端到端冒烟测试: 用合成数据跑 1 个 epoch 训练 + 验证 + 保存 + 推理.

不追求指标, 只验证整条流水线 (数据 -> 模型 -> 损失 -> 反传 -> 评估 -> ckpt
-> RegionAnalyzer / FabricPredictor) 能跑通且不产生 NaN.
"""

from pathlib import Path

import numpy as np
import torch

from fabric_ncd.engine import Trainer
from fabric_ncd.inference import FabricPredictor
from fabric_ncd.utils import load_yaml
from fabric_ncd.data.demo import make_dataset


def _config(root: Path, out_dir: Path) -> dict:
    cfg = load_yaml("configs/fabric19.yaml")
    cfg["experiment"].update({"output_dir": str(out_dir), "device": "cpu", "amp": False, "seed": 0})
    cfg["data"].update(
        {
            "dataset_yaml": str(root / "dataset.yaml"),
            "image_size": [64, 64],
            "batch_size": 3,
            "eval_batch_size": 2,
            "num_workers": 0,
            "normal_ratio": 0.5,
            "limit_train": 8,
            "limit_val": 4,
        }
    )
    cfg["data"]["augment"].update({"synth_prob": 0.5, "weave_norm": True, "width_margin": 0.3})
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["model"]["heads"]["num_classes"] = 3
    cfg["model"]["loss"]["fabric"]["edge_weight"] = 0.2
    cfg["train"].update(
        {
            "epochs": 1,
            "warmup_epochs": 0,
            "log_interval": 1,
            "memory_warmup_batches": 1,
            "memory_refresh_interval": 0,
            "ema": 0.9,
            "val_interval": 1,
        }
    )
    cfg["eval"].update({"threshold": 0.5, "min_area_ratio": 0.001})
    return cfg


def test_end_to_end(tmp_path):
    root = make_dataset(tmp_path / "data", n_train=6, n_val=4, n_normal=3, size=(64, 64))
    out_dir = tmp_path / "run"
    cfg = _config(root, out_dir)

    trainer = Trainer(cfg)
    summary = trainer.fit()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()
    assert summary["history"]
    metrics = summary.get("final_metrics", {})
    assert "num_samples" in metrics
    assert torch.isfinite(torch.tensor(metrics.get("pixel_auroc", 0.5)))

    predictor = FabricPredictor(checkpoint=out_dir / "best.pt", config=cfg, device="cpu")
    from fabric_ncd.data.demo import weave_image

    image = weave_image((64, 64), np.random.default_rng(0))
    result = predictor.predict(image)
    assert "regions" in result
    assert result["image"]["image_score"] >= 0
    out_png = out_dir / "overlay.png"
    predictor.save_overlay(image, result, out_png)
    assert out_png.exists()
