"""Single-GPU/CPU training and evaluation for the Fabric19 adaptation."""
from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import class_order, load_class_names
from .data import FabricRegions, load_manifest
from .losses import Objectives
from .metrics import classification_metrics, image_metrics, merge_image_probabilities
from .model import get_params_groups, load_model
from .transforms import PairTransform, TwoViews, worker_init_fn

LOG = logging.getLogger("anomalyncd")
CHECKPOINT_VERSION = 1


def setup_logging(output: Path | None = None) -> None:
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output / "train.log", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        LOG.addHandler(handler)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 检测不到 GPU；使用 --device cpu 可运行小型测试")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def make_loader(dataset: FabricRegions, cfg: dict, training: bool, device: torch.device,
                generator: torch.Generator | None = None) -> DataLoader:
    sampler = None
    if training:
        labels = dataset.training_labels
        if cfg["mode"] == "discovery":
            # Binary known/unknown membership only; do not balance by novel class ID.
            known_count = sum(x >= 0 for x in labels)
            novel_count = len(labels)-known_count
            counts = Counter(x for x in labels if x >= 0)
            weights = []
            for label in labels:
                if label < 0:
                    weights.append(1.0/novel_count)
                elif cfg["training"]["class_balanced_sampler"]:
                    weights.append(1.0/counts[label])
                else:
                    weights.append(1.0/known_count)
        elif cfg["training"]["class_balanced_sampler"]:
            counts = Counter(labels)
            weights = [1.0/counts[label] for label in labels]
        else:
            weights = [1.0] * len(labels)
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(labels), replacement=True, generator=generator)
    workers = cfg["training"]["num_workers"]
    # Opt-in. Reusing workers avoids re-spawning them every epoch (with spawn on
    # Windows that dominated wall time: ~44s of a 54s epoch). It is OFF by
    # default because worker RNG is seeded once instead of per epoch, so the
    # augmentation stream on a resumed run differs from a non-persistent run.
    # Sampling ORDER is unaffected: it comes from the main-process generator.
    persistent = bool(cfg["training"].get("persistent_workers", False)) and workers > 0
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=training and sampler is None,
                      sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda",
                      drop_last=False, persistent_workers=persistent,
                      worker_init_fn=worker_init_fn, generator=generator,
                      multiprocessing_context="spawn" if workers and workers > 0 else None)


def validate_prepared(cfg: dict, root: str | Path) -> dict:
    names = load_class_names(cfg["data"]["dataset_config"])
    manifest = load_manifest(root, names)
    expected = {"padding": cfg["data"]["padding"], "min_crop_size": cfg["data"]["min_crop_size"], "mask_source": "yolo_bbox"}
    if manifest["crop"] != expected:
        raise ValueError("训练 crop 参数与预处理不一致；请按当前配置重新运行 prepare_data.py")
    return manifest


def model_signature(cfg: dict, order: list[int], n_known: int, names: list[str], fingerprint: str) -> dict:
    keys = ("backbone", "image_size", "grad_from_block", "mask_layers", "n_heads", "n_views", "hidden_dim", "bottleneck_dim")
    return {"model_config": {k: cfg["model"][k] for k in keys}, "mode": cfg["mode"], "class_order": order,
            "n_known": n_known, "names": names, "dataset_fingerprint": fingerprint}


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("不是本项目的 Fabric19 checkpoint；原版 AnomalyNCD 权重的类别空间不兼容")
    return checkpoint


def checkpoint_cfg(checkpoint: dict) -> dict:
    return checkpoint["config"]


def checkpoint_dataset(checkpoint: dict, root: str | Path, transform=None, evaluation: bool = True) -> FabricRegions:
    manifest = load_manifest(root, checkpoint["names"])
    if manifest["fingerprint"] != checkpoint["dataset_fingerprint"]:
        raise ValueError("评估数据与 checkpoint 的预处理指纹不一致；请恢复原数据或重新训练")
    return FabricRegions(root, "val", checkpoint["class_order"], checkpoint["n_known"], checkpoint["mode"],
                         transform=transform, evaluation=evaluation)


def selected_head(checkpoint: dict) -> int:
    sums, counts = checkpoint["head_loss_sum"], checkpoint["head_loss_count"]
    values = [s/c if c else math.inf for s, c in zip(sums, counts, strict=True)]
    head = int(np.argmin(values))
    if not math.isfinite(values[head]):
        raise ValueError("Checkpoint 没有有效的训练 head loss，无法按训练指标选择 head")
    return head


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   head: int, temperature: float, amp: bool = False, desc: str = "predict") -> np.ndarray:
    model.eval()
    result = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["images"][0].to(device, non_blocking=device.type == "cuda")
        masks = batch["masks"][0].to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            _, logits = model(images, masks)
        probs = torch.softmax(logits[head].float() / temperature, dim=-1)
        if not bool(torch.isfinite(probs).all()):
            raise FloatingPointError("模型输出 NaN/Inf")
        result.append(probs.cpu().numpy())
    return np.concatenate(result, axis=0)


def write_predictions(output: Path, dataset: FabricRegions, probs: np.ndarray, cfg: dict,
                      head: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    order, names = dataset.class_order, dataset.manifest["names"]
    predictions = probs.argmax(axis=1)
    targets = np.asarray(dataset.labels)
    def region_summary(selected, label_ids=None):
        result = classification_metrics(targets[selected], probs[selected])
        label_ids = order if label_ids is None else label_ids
        if "per_class" in result:
            result["per_class"] = {str(label_ids[int(k)]): {"name": names[label_ids[int(k)]], **v}
                                   for k, v in result["per_class"].items()}
        return result

    metrics = {"selected_head": head, "head_selection": "training_loss_only",
               "class_order": order, "class_names": [names[i] for i in order],
               "probability_column_order": "class_order",
               "region_all": region_summary(np.ones(len(targets), dtype=bool)), "label_space": "all_classes_no_gt_gating"}
    if cfg["mode"] == "discovery":
        n_known = dataset.n_known
        for name, subset in (("known", targets < n_known), ("novel", targets >= n_known)):
            metrics[f"region_{name}_subset"] = region_summary(subset) if subset.any() else {"count": 0}
        # Explicit closed-set diagnostic, NOT the primary deployment metric.
        subset = targets >= n_known
        if subset.any():
            novel_probs = probs[subset, n_known:]
            novel_probs = novel_probs / novel_probs.sum(axis=1, keepdims=True)
            remapped = {label: index for index, label in enumerate(range(n_known, len(order)))}
            novel_targets = np.array([remapped[int(x)] for x in targets[subset]])
            diagnostic = classification_metrics(novel_targets, novel_probs)
            diagnostic["local_class_order"] = order[n_known:]
            diagnostic["per_class"] = {
                str(order[n_known:][int(k)]): {"name": names[order[n_known:][int(k)]], **v}
                for k, v in diagnostic["per_class"].items()
            }
            metrics["region_novel_closed_set_diagnostic"] = diagnostic
    image_probs, image_targets = merge_image_probabilities(dataset.records, probs, cfg["inference"]["area_temperature"])
    image_probs = np.asarray(image_probs)
    metrics["image_all"], image_labels = image_metrics(image_targets, image_probs, cfg["inference"]["image_threshold"])
    metrics["evaluation_note"] = (
        "YOLO boxes define evaluation regions; this is not end-to-end detection. "
        "NCD primary metrics use all logits without ground-truth known/novel gating. "
        "The novel closed-set diagnostic assumes a novel region and is not a deployment score."
    )
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "region_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "image_id", "source", "target_id", "target_name", "prediction_id", "prediction_name", "confidence", "class_probabilities"])
        for index, record in enumerate(dataset.records):
            target, pred = order[int(targets[index])], order[int(predictions[index])]
            writer.writerow([record["sample_id"], record["image_id"], record["source"], target, names[target],
                             pred, names[pred], float(probs[index].max()), json.dumps(probs[index].tolist())])
    with (output / "image_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "source", "target_ids", "target_names", "prediction_ids", "prediction_names", "class_probabilities"])
        seen = {}
        for record in dataset.records:
            seen.setdefault(record["image_id"], record["source"])
        for index, (image_id, source) in enumerate(seen.items()):
            target_ids = [order[c] for c in image_targets[index]]
            pred_ids = [order[c] for c in image_labels[index]]
            writer.writerow([image_id, source, json.dumps(target_ids), json.dumps([names[c] for c in target_ids], ensure_ascii=False),
                             json.dumps(pred_ids), json.dumps([names[c] for c in pred_ids], ensure_ascii=False),
                             json.dumps(image_probs[index].tolist())])
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, epoch, device, head_sums, head_counts) -> dict:
    model.train()
    meters, n_samples = Counter(), 0
    steps = cfg["training"]["accumulation_steps"]
    amp = cfg["training"]["amp"] and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"epoch {epoch+1}/{cfg['training']['epochs']}", leave=False)
    for step, batch in enumerate(progress):
        images = torch.cat(batch["images"], dim=0).to(device, non_blocking=device.type == "cuda")
        masks = torch.cat(batch["masks"], dim=0).to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device)
        labelled = batch["labelled"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            projection, logits = model(images, masks)
            loss, components, head_losses = criterion(projection, logits, targets, labelled, epoch)
        if not math.isfinite(float(loss.detach())) or not all(math.isfinite(x) for x in head_losses):
            raise FloatingPointError(f"epoch {epoch}, step {step}: non-finite loss; lower LR or disable AMP")
        scaler.scale(loss/steps).backward()
        if (step+1) % steps == 0 or step+1 == len(loader):
            if cfg["training"]["max_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = targets.numel()
        n_samples += batch_size
        for key, value in components.items():
            meters[key] += value*batch_size
        for index, value in enumerate(head_losses):
            head_sums[index] += value*batch_size
            head_counts[index] += batch_size
        progress.set_postfix(loss=f"{components['total']:.4f}")
        if step % cfg["experiment"]["print_freq"] == 0:
            LOG.info("epoch=%d step=%d/%d %s", epoch+1, step+1, len(loader),
                     " ".join(f"{k}={v:.4f}" for k, v in components.items()))
    return {key: value/n_samples for key, value in meters.items()}


def train(cfg: dict, resume: str | Path | None = None, allow_random_init: bool = False) -> dict:
    output = Path(cfg["experiment"]["output_dir"]).expanduser().resolve()
    if resume:
        resume = Path(resume).expanduser().resolve()
        if output != resume.parent:
            raise ValueError("续训请保持原 output_dir，--resume 指向该目录的 last.pt")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"实验目录非空: {output}；请换目录或用 --resume 续训，避免覆盖结果")
    setup_logging(output)
    seed_everything(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])
    manifest = validate_prepared(cfg, cfg["data"]["prepared_root"])
    names = load_class_names(cfg["data"]["dataset_config"])
    nc = len(names)
    known_ids = cfg["discovery"]["known_class_ids"] if cfg["mode"] == "discovery" else None
    order, _ = class_order(nc, cfg["mode"], known_ids)
    n_known = len(known_ids) if known_ids else nc
    signature = model_signature(cfg, order, n_known, names, manifest["fingerprint"])
    size = cfg["model"]["image_size"]
    train_data = FabricRegions(cfg["data"]["prepared_root"], "train", order, n_known, cfg["mode"],
                               TwoViews(PairTransform(size, training=True)))
    val_data = FabricRegions(cfg["data"]["prepared_root"], "val", order, n_known, cfg["mode"],
                             PairTransform(size, training=False), evaluation=True)
    generator = torch.Generator().manual_seed(cfg["experiment"]["seed"])
    train_loader = make_loader(train_data, cfg, True, device, generator)
    val_loader = make_loader(val_data, cfg, False, device)
    model = load_model(cfg, nc, load_pretrained=resume is None, allow_random_init=allow_random_init).to(device)
    criterion = Objectives(cfg, nc, n_known)
    optimizer = torch.optim.SGD(get_params_groups(model), lr=cfg["training"]["lr"],
                                momentum=cfg["training"]["momentum"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"], eta_min=cfg["training"]["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
    start_epoch, best_metric = 0, -math.inf
    random_init = allow_random_init
    head_sums, head_counts = [0.0]*cfg["model"]["n_heads"], [0]*cfg["model"]["n_heads"]
    history = []
    if resume:
        checkpoint = load_checkpoint(resume)
        saved_signature = {key: checkpoint[key] for key in signature}
        # Model and data must match; optimizer schedule is restored, not reconfigured.
        if saved_signature != signature or checkpoint["config"]["training"] != cfg["training"] or checkpoint["config"]["loss"] != cfg["loss"]:
            raise ValueError("续训的模型、类别划分、数据指纹、training 或 loss 配置与 checkpoint 不一致")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_metric = checkpoint["epoch"], checkpoint["best_metric"]
        random_init = bool(checkpoint.get("allow_random_init", False))
        head_sums, head_counts = checkpoint["head_loss_sum"], checkpoint["head_loss_count"]
        history = checkpoint.get("history", [])
        generator.set_state(checkpoint["generator"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        np_state = checkpoint["numpy_rng"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
        if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    LOG.info("mode=%s device=%s train_regions=%d val_regions=%d classes=%d known=%d", cfg["mode"], device,
             len(train_data), len(val_data), nc, n_known)
    LOG.info("output index -> YOLO class: %s", {i: names[c] for i, c in enumerate(order)})
    if random_init and cfg["model"]["backbone"] != "tiny_vit":
        LOG.warning("EXPERIMENT ONLY: random backbone initialization; not a pretrained-paper result")
    LOG.info("Normal images are audited but excluded from the 19-class region objective.")
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        began = time.monotonic()
        losses = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg, epoch, device, head_sums, head_counts)
        scheduler.step()
        state = {"format_version": CHECKPOINT_VERSION, **signature, "config": cfg, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch+1, "best_metric": best_metric, "head_loss_sum": head_sums, "head_loss_count": head_counts,
                 "history": history, "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                 "python_rng": random.getstate(), "numpy_rng": [np.random.get_state()[0], np.random.get_state()[1].tolist(),
                                                            *np.random.get_state()[2:]],
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "allow_random_init": random_init}
        state["selected_head"] = selected_head(state)
        eval_every = cfg["training"]["eval_every"]
        if eval_every and ((epoch+1) % eval_every == 0 or epoch+1 == cfg["training"]["epochs"]):
            probs = predict_loader(model, val_loader, device, state["selected_head"], cfg["loss"]["temperature"], cfg["training"]["amp"])
            metrics = write_predictions(output / "validation", val_data, probs, cfg, state["selected_head"])
            score = metrics["region_all"]["macro_f1"]
            history.append({"epoch": epoch+1, "train": losses, "validation_macro_f1": score})
            LOG.info("epoch=%d val_region_macro_f1=%.4f val_image_micro_f1=%.4f", epoch+1, score, metrics["image_all"]["micro_f1"])
            if cfg["mode"] == "supervised" and score > best_metric:
                best_metric = score
                state["best_metric"] = best_metric
                torch.save(state, output / "best.tmp.pt")
                (output / "best.tmp.pt").replace(output / "best.pt")
        else:
            history.append({"epoch": epoch+1, "train": losses})
        state["best_metric"] = best_metric
        torch.save(state, output / "last.tmp.pt")
        (output / "last.tmp.pt").replace(output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOG.info("epoch=%d finished in %.1fs, last.pt saved", epoch+1, time.monotonic()-began)
    # NCD never selects best.pt using validation. Supervised uses its recorded best.
    final_path = output / ("best.pt" if cfg["mode"] == "supervised" and (output / "best.pt").exists() else "last.pt")
    final_checkpoint = load_checkpoint(final_path)
    model.load_state_dict(final_checkpoint["model"], strict=True)
    head = selected_head(final_checkpoint)
    probs = predict_loader(model, val_loader, device, head, cfg["loss"]["temperature"], cfg["training"]["amp"], "final validation")
    metrics = write_predictions(output / "final_evaluation", val_data, probs, cfg, head)
    metrics["checkpoint"] = str(final_path)
    metrics["epoch"] = final_checkpoint["epoch"]
    (output / "final_evaluation" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOG.info("完成。checkpoint=%s region_macro_f1=%.4f image_micro_f1=%.4f", final_path,
             metrics["region_all"]["macro_f1"], metrics["image_all"]["micro_f1"])
    return metrics


def evaluate(checkpoint_path: str | Path, prepared_root: str | Path | None = None,
             output: str | Path | None = None, device_name: str = "auto", config: dict | None = None) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = config if config is not None else checkpoint_cfg(checkpoint)
    root = Path(prepared_root or cfg["data"]["prepared_root"])
    output = Path(output) if output else Path(checkpoint_path).resolve().parent / "evaluation"
    setup_logging(output)
    device = resolve_device(device_name)
    data = checkpoint_dataset(checkpoint, root, PairTransform(cfg["model"]["image_size"], training=False))
    loader = make_loader(data, cfg, False, device)
    model = load_model(cfg, len(checkpoint["class_order"]), load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    head = selected_head(checkpoint)
    probs = predict_loader(model, loader, device, head, cfg["loss"]["temperature"], cfg["training"]["amp"], "evaluate")
    metrics = write_predictions(output, data, probs, cfg, head)
    LOG.info("评估结果: %s", output / "metrics.json")
    return metrics
