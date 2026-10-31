from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

from anomalyncd.config import deep_merge, load_config
from anomalyncd.data import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """19-class synthetic YOLO data; visual accuracy on this data is meaningless."""
    base = tmp_path_factory.mktemp("fabric19")
    raw, prepared = base / "raw", base / "prepared"
    for split in ("train", "val"):
        (raw / "images" / split).mkdir(parents=True)
        (raw / "labels" / split).mkdir(parents=True)
        for class_id in range(19):
            relative = Path("布卷.01") / f"缺陷.{class_id:02d}.png"
            image_path, label_path = raw / "images" / split / relative, raw / "labels" / split / relative.with_suffix(".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (64, 48), (25+class_id*8, 70, 130))
            draw = ImageDraw.Draw(image)
            # Ensure train/val are distinct files, as real split data must be.
            draw.point((63, 47), fill=(255, 255, 255) if split == "val" else (0, 0, 0))
            x = 5 + class_id % 12
            draw.rectangle((x, 8, x+8, 16), fill=(220, 40+class_id*5, 60))
            if class_id == 0:
                draw.rectangle((43, 30, 53, 39), fill=(240, 230, 30))
            image.save(image_path)
            lines = [f"{class_id} {(x+4)/64} 0.25 {9/64} {9/48}"]
            if class_id == 0:
                lines.append("7 0.75 0.71875 0.15625 0.2083333333")
            label_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for split in ("train", "val"):
        normal = raw / "normal" / "images" / split
        normal.mkdir(parents=True)
        normal_image = Image.new("RGB", (32, 32), "gray")
        ImageDraw.Draw(normal_image).point((31, 31), fill="white" if split == "val" else "black")
        normal_image.save(normal / "good.png")
    dataset_yaml = base / "fabric19.yaml"
    original = yaml.safe_load((ROOT / "dataset/fabric19.yaml").read_text(encoding="utf-8"))
    dataset_yaml.write_text(yaml.safe_dump(original, allow_unicode=True), encoding="utf-8")
    manifest = prepare_dataset(dataset_yaml, prepared, data_root=raw, progress=lambda _: None)
    return {"base": base, "raw": raw, "prepared": prepared, "yaml": dataset_yaml, "manifest": manifest}


@pytest.fixture
def smoke_config(synthetic, tmp_path):
    cfg = load_config(ROOT / "configs/fabric19.yaml")
    cfg = deep_merge(cfg, {
        "data": {"dataset_config": str(synthetic["yaml"]), "prepared_root": str(synthetic["prepared"])},
        "model": {"backbone": "tiny_vit", "image_size": 32, "grad_from_block": 3, "mask_layers": 2,
                  "n_heads": 2, "hidden_dim": 32, "bottleneck_dim": 16},
        "training": {"batch_size": 5, "num_workers": 0, "epochs": 2, "accumulation_steps": 2,
                     "amp": False, "eval_every": 1, "lr": 0.001, "min_lr": 0.00001},
        "loss": {"warmup_teacher_temp_epochs": 1},
        # tiny_vit has depth 4 and patch 8: the production anomaly defaults
        # (layers 3,5,7 @ 448px) are invalid here and must be scaled down too.
        "anomaly": {"enabled": True, "image_size": 32, "layers": [1, 2, 3], "neighborhood": 3,
                    "bank_size": 64, "max_candidates": 512, "coreset": True, "knn": 3,
                    "smooth_sigma": 1.0, "topk_ratio": 0.05, "min_area": 4,
                    "max_regions": 5, "threshold_steps": 12},
        "experiment": {"device": "cpu", "output_dir": str(tmp_path / "run"), "print_freq": 10},
    })
    return cfg
