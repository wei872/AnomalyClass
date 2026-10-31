import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from anomalyncd.data import prepare_dataset
from anomalyncd.detector import prepare_detector_dataset
from anomalyncd.split_audit import audit_split, difference_hash, quarantine_leaked_val, restore_quarantine


@pytest.fixture
def leaky_dataset(synthetic, tmp_path):
    """Copy the synthetic dataset and make 3 val images byte-identical to train."""
    raw = tmp_path / "leaky"
    shutil.copytree(synthetic["raw"], raw)
    for class_id in (0, 1, 2):
        train = raw / "images/train/布卷.01" / f"缺陷.{class_id:02d}.png"
        val = raw / "images/val/布卷.01" / f"缺陷.{class_id:02d}.png"
        val.write_bytes(train.read_bytes())
    return raw, synthetic["yaml"]


def test_audit_reports_exact_cross_split_duplicates(leaky_dataset):
    raw, yaml_path = leaky_dataset
    report = audit_split(yaml_path, raw)
    assert report["exact_cross_split_count"] == 3
    assert len(report["leaked_val_images"]) == 3
    assert all(key.startswith("val/defect/") for key in report["leaked_val_images"])
    assert report["counts"]["train_defect"] == 19


def test_drop_leaked_val_keeps_originals_and_shrinks_val(leaky_dataset, tmp_path):
    raw, yaml_path = leaky_dataset
    before = {p.relative_to(raw).as_posix(): p.read_bytes()
              for p in (raw / "images/val").rglob("*.png")}
    manifest = prepare_dataset(yaml_path, tmp_path / "regions", raw,
                               drop_leaked_val=True, progress=lambda _: None)
    assert manifest["stats"]["dropped_leaked_val"] == 3
    assert manifest["stats"]["val"]["images"] == 16
    sources = {r["source"] for r in manifest["splits"]["val"]}
    assert not any(f"缺陷.{c:02d}.png" in s for s in sources for c in (0, 1, 2))
    after = {p.relative_to(raw).as_posix(): p.read_bytes()
             for p in (raw / "images/val").rglob("*.png")}
    assert before == after, "原始 val 图片不能被修改"


def test_detector_drop_leaked_val_matches_region_policy(leaky_dataset, tmp_path):
    raw, yaml_path = leaky_dataset
    manifest = prepare_detector_dataset(yaml_path, tmp_path / "detector", raw,
                                        drop_leaked_val=True, progress=lambda _: None)
    assert manifest["stats"]["dropped_leaked_val"] == 3
    assert manifest["stats"]["val"]["images"] == 16
    assert manifest["stats"]["val"]["total_images"] == 17  # 16 defect + 1 normal
    train_hashes = {r["sha256"] for r in manifest["splits"]["train"] if r["kind"] == "defect"}
    val_hashes = {r["sha256"] for r in manifest["splits"]["val"] if r["kind"] == "defect"}
    assert not (train_hashes & val_hashes)


def test_default_still_rejects_and_flags_conflict(leaky_dataset, tmp_path):
    raw, yaml_path = leaky_dataset
    with pytest.raises(ValueError, match="--drop-leaked-val"):
        prepare_detector_dataset(yaml_path, tmp_path / "d1", raw, progress=lambda _: None)
    with pytest.raises(ValueError, match="不能同时使用"):
        prepare_dataset(yaml_path, tmp_path / "d2", raw, allow_overlap=True,
                        drop_leaked_val=True, progress=lambda _: None)


def test_allow_overlap_still_works(leaky_dataset, tmp_path):
    raw, yaml_path = leaky_dataset
    manifest = prepare_detector_dataset(yaml_path, tmp_path / "detector", raw,
                                        allow_overlap=True, progress=lambda _: None)
    assert manifest["stats"]["exact_train_val_overlap"] == 3
    assert manifest["stats"]["val"]["images"] == 19
    assert manifest["stats"]["val"]["total_images"] == 20


def test_quarantine_dry_run_moves_nothing_then_apply_and_restore(leaky_dataset, tmp_path):
    raw, yaml_path = leaky_dataset
    destination = tmp_path / "quarantine"
    preview = quarantine_leaked_val(yaml_path, destination, raw, apply=False, progress=lambda _: None)
    assert preview["count"] == 3 and not destination.exists()
    assert audit_split(yaml_path, raw)["exact_cross_split_count"] == 3

    applied = quarantine_leaked_val(yaml_path, destination, raw, apply=True, progress=lambda _: None)
    assert applied["count"] == 3
    assert audit_split(yaml_path, raw)["exact_cross_split_count"] == 0
    for record in applied["files"]:
        assert not Path(record["image"]).exists()

    restore_quarantine(destination / "quarantine.json", progress=lambda _: None)
    assert audit_split(yaml_path, raw)["exact_cross_split_count"] == 3


def test_quarantine_refuses_directory_inside_dataset(leaky_dataset, tmp_path):
    raw, yaml_path = leaky_dataset
    with pytest.raises(ValueError, match="隔离目录不能位于数据目录内"):
        quarantine_leaked_val(yaml_path, raw / "images" / "quarantine", raw, progress=lambda _: None)


def test_perceptual_hash_finds_augmented_near_duplicates(synthetic, tmp_path):
    raw = tmp_path / "aug"
    (raw / "images/train").mkdir(parents=True)
    (raw / "images/val").mkdir(parents=True)
    (raw / "labels/train").mkdir(parents=True)
    (raw / "labels/val").mkdir(parents=True)
    base = Image.new("RGB", (64, 64), (40, 90, 150))
    draw = ImageDraw.Draw(base)
    draw.rectangle((12, 12, 34, 34), fill=(230, 60, 70))
    base.save(raw / "images/train/a.png")
    # Brightness-only augmentation: different bytes, same source image.
    Image.eval(base, lambda v: min(255, int(v * 1.12))).save(raw / "images/val/a_aug.png")
    for split, name in (("train", "a"), ("val", "a_aug")):
        (raw / f"labels/{split}/{name}.txt").write_text("0 0.36 0.36 0.34 0.34\n", encoding="utf-8")
    yaml_path = synthetic["yaml"]
    exact = audit_split(yaml_path, raw)
    assert exact["exact_cross_split_count"] == 0
    near = audit_split(yaml_path, raw, perceptual=True, max_distance=5)
    assert near["near_cross_split_count"] >= 1
    assert near["near_cross_split"][0]["val"].endswith("a_aug.png")


def test_dhash_is_stable_and_sensitive():
    a = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(a).rectangle((4, 4, 20, 20), fill="black")
    b = Image.eval(a, lambda v: min(255, int(v * 1.1)))
    c = Image.new("RGB", (32, 32), "gray")
    assert difference_hash(a) == difference_hash(b)
    assert (difference_hash(a) ^ difference_hash(c)).bit_count() > 5
