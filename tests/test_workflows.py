"""真实命令行回归：checkpoint 配置、样本数、切块和实时基准。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fabric_ncd.data.demo import make_dataset
from fabric_ncd.models.fabric_ncd import FabricNCD
from fabric_ncd.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trained_fixture(tmp_path):
    root = make_dataset(tmp_path / "data", n_train=4, n_val=4, n_normal=3, size=(64, 96))
    cfg = load_yaml(ROOT / "configs/fabric19.yaml")
    cfg["data"].update(dataset_yaml=str(root / "dataset.yaml"), image_size=[64, 96],
                       batch_size=2, eval_batch_size=2, num_workers=0, cloth_mask="full")
    cfg["data"]["augment"]["weave_norm"] = False
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["model"]["heads"]["num_classes"] = 3
    cfg["model"]["memory"]["enabled"] = False
    cfg["infer"].update(weave_norm=False, fp16=False, out_stride=4)
    cfg["experiment"].update(device="cpu", amp=False)
    model = FabricNCD(cfg).eval()
    path = tmp_path / "model.pt"
    torch.save({"config": cfg, "model": model.state_dict(), "names": ["a", "b", "c"]}, path)
    image = tmp_path / "image.png"
    Image.fromarray(np.full((64, 96, 3), 128, np.uint8)).save(image)
    return path, image, root, cfg


def run_tool(*args):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


def test_infer_uses_checkpoint_config_by_default(trained_fixture, tmp_path):
    checkpoint, image, root, _ = trained_fixture
    out = tmp_path / "pred"
    run_tool("tools/infer.py", "--ckpt", checkpoint, "--source", image,
             "--device", "cpu", "--out", out, "--no-overlay")
    payload = json.loads((out / "results.json").read_text())
    assert str(image) in payload["results"]


def test_evaluate_honors_max_samples_and_tiled_mode(trained_fixture, tmp_path):
    checkpoint, _, root, _ = trained_fixture
    output = tmp_path / "eval.json"
    run_tool("tools/evaluate.py", "--ckpt", checkpoint, "--root", root, "--device", "cpu",
             "--batch-size", 2, "--max-samples", 3, "--save-preds", output)
    result = json.loads(output.read_text())
    assert result["metrics"]["num_samples"] == 3
    assert len(result["predictions"]) == 3

    output_tile = tmp_path / "eval_tiled.json"
    run_tool("tools/evaluate.py", "--ckpt", checkpoint, "--root", root, "--device", "cpu",
             "--set", "eval.tile=true", "--set", "eval.tile_size=[32,48]",
             "--set", "eval.tile_overlap=8", "--max-samples", 1, "--save-preds", output_tile)
    result_tile = json.loads(output_tile.read_text())
    assert result_tile["metrics"]["num_samples"] == 1
    assert result_tile["metrics"]["tile_size"] == [32, 48]


def test_realtime_benchmark_accepts_root_and_overrides_without_extra_frames(trained_fixture, tmp_path):
    checkpoint, image, root, _ = trained_fixture
    output = tmp_path / "bench.json"
    run_tool("tools/rt_bench.py", "--ckpt", checkpoint, "--root", root, "--source", image,
             "--device", "cpu", "--frames", 4, "--warmup", 2, "--set", "infer.weave_norm=false",
             "--json", output)
    result = json.loads(output.read_text())
    assert result["frames"] == 4
    assert result["fps_wall"] > 0


def test_onnx_export_uses_rectangular_training_size(trained_fixture, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    checkpoint, _, _, _ = trained_fixture
    out = tmp_path / "model.onnx"
    run_tool("tools/export_onnx.py", "--ckpt", checkpoint, "--out", out, "--check")
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == ["batch", 3, 64, 96]


def test_dynamic_onnx_accepts_another_resolution(trained_fixture, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    checkpoint, _, _, cfg = trained_fixture
    out = tmp_path / "dynamic.onnx"
    run_tool("tools/export_onnx.py", "--ckpt", checkpoint, "--out", out, "--dynamic", "--check")
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    image = np.random.default_rng(11).normal(size=(2, 3, 96, 128)).astype(np.float32)
    desc = np.zeros((2, 6), np.float32)
    actual = session.run(None, {"images": image, "weave_desc": desc})
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FabricNCD(cfg).eval()
    model.load_state_dict(state["model"])
    with torch.no_grad():
        output = model(torch.from_numpy(image), compute_memory=False, weave_desc=torch.from_numpy(desc))
    keys = ["anomaly_logit", "cloth_logit", "boundary_logit", "class_logits", "embedding"]
    for value, key in zip(actual, keys):
        np.testing.assert_allclose(value, output[key].numpy(), rtol=2e-3, atol=2e-4)


def test_training_cli_can_resume_using_saved_configuration(trained_fixture, tmp_path):
    from fabric_ncd.utils import save_yaml

    _, _, _, cfg = trained_fixture
    out = tmp_path / "training"
    cfg["experiment"]["output_dir"] = str(out)
    cfg["train"].update(epochs=1, warmup_epochs=0, memory_warmup_batches=1,
                        memory_refresh_interval=0, scale_sweep=[], log_interval=100)
    cfg["data"].update(limit_train=4, limit_val=4)
    cfg["eval"]["threshold"] = 0.5
    config_file = tmp_path / "train.yaml"
    save_yaml(cfg, config_file)
    run_tool("tools/train.py", "--config", config_file)
    first = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 0 and first["global_step"] > 0
    run_tool("tools/train.py", "--resume", out / "last.pt", "--epochs", 2)
    resumed = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 1
    assert resumed["global_step"] > first["global_step"]
    assert resumed["ema_updates"] > first["ema_updates"]


def test_tiled_evaluation_matches_predictor_for_the_same_image(trained_fixture):
    from fabric_ncd.data.yolo import list_images, load_dataset_spec, load_rgb
    from fabric_ncd.engine import evaluate_tiled
    from fabric_ncd.inference import FabricPredictor

    checkpoint, _, _, _ = trained_fixture
    overrides = {"infer": {"tile": True, "tile_size": [32, 48], "tile_overlap": 8,
                           "out_stride": 3, "probs_mul": 2, "weave_norm": False},
                 "eval": {"tile": True, "tile_size": [32, 48], "tile_overlap": 8,
                          "analysis_stride": 3, "threshold": 0.1, "cloth_gate": False}}
    predictor = FabricPredictor(checkpoint=checkpoint, config=overrides, device="cpu")
    image = list_images(load_dataset_spec(predictor.cfg).splits["val"])[0]
    prediction = predictor.predict(load_rgb(image))
    metrics = evaluate_tiled(predictor.model, predictor.cfg, torch.device("cpu"),
                             max_samples=1, collect_predictions=True)
    evaluated = metrics["predictions"][0]
    assert evaluated["image_score"] == pytest.approx(prediction["image"]["image_score"], abs=1e-6)
    assert len(evaluated["regions"]) == len(prediction["regions"])
    for a, b in zip(evaluated["regions"], prediction["regions"]):
        assert a["bbox"] == pytest.approx(b["bbox"], abs=1e-6)
        assert a["score"] == pytest.approx(b["score"], abs=1e-6)


def test_infer_rectangular_tiles_and_preserves_duplicate_filenames(trained_fixture, tmp_path):
    checkpoint, image, _, _ = trained_fixture
    source, out = tmp_path / "source", tmp_path / "overlays"
    for folder in ("a", "b"):
        target = source / folder / "same.png"
        target.parent.mkdir(parents=True)
        target.write_bytes(image.read_bytes())
    run_tool("tools/infer.py", "--ckpt", checkpoint, "--source", source, "--device", "cpu",
             "--tile", "--tile-size", "32x48", "--tile-overlap", 8, "--out", out)
    assert (out / "a" / "same_pred.png").is_file()
    assert (out / "b" / "same_pred.png").is_file()


def test_bench_accounts_for_batch_and_memory(trained_fixture, tmp_path):
    output = tmp_path / "bench_model.json"
    run_tool("tools/bench.py", "--size", 64, "--device", "cpu", "--batch-size", 2, "--iters", 1,
             "--set", "model.memory.capacity=32", "--out", output)
    result = json.loads(output.read_text())
    assert result["batch_size"] == 2
    assert result["memory_bank_size"] == 32
    assert result["macs_memory_G"] > 0
    assert result["fps"] > 0


def test_prepare_dataset_exports_nested_same_name_masks(trained_fixture, tmp_path):
    _, image, root, _ = trained_fixture
    for folder, cls in (("a", 0), ("b", 1)):
        image_path = root / "images" / "train" / folder / "same.png"
        label_path = root / "labels" / "train" / folder / "same.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image.read_bytes())
        label_path.write_text(f"{cls} 0.5 0.5 0.25 0.25\n")
    out = tmp_path / "defect_masks"
    run_tool("tools/prepare_dataset.py", "--set", f"data.dataset_yaml={root / 'dataset.yaml'}",
             "--export-defect-masks", "--defect-mask-out", out, "--workers", 1, "--mask-size", 64)
    assert (out / "train" / "a" / "same.png").is_file()
    assert (out / "train" / "b" / "same.png").is_file()


def test_diagnostics_honors_exact_sample_limit(trained_fixture, tmp_path):
    import csv

    checkpoint, _, root, _ = trained_fixture
    output = tmp_path / "scores.csv"
    run_tool("tools/diagnose_scores.py", "--checkpoint", checkpoint, "--root", root, "--device", "cpu",
             "--samples", 3, "--memory-batches", 1, "--csv", output)
    with output.open() as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_discover_workflow_saves_clusters_and_prototypes(trained_fixture, tmp_path):
    checkpoint, _, root, _ = trained_fixture
    out = tmp_path / "discovery"
    run_tool("tools/discover.py", "--ckpt", checkpoint, "--source", root / "images" / "val",
             "--out", out, "--device", "cpu", "--set", "eval.threshold=0.01",
             "--set", "eval.cloth_gate=false", "--k", 2, "--max-images", 3)
    report = json.loads((out / "ncd_report.json").read_text())
    assert report["num_regions"] >= 1
    assert (out / "discovered_prototypes.pt").is_file()
