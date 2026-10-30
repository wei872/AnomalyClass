"""损失函数.

* 异常分割: Focal + Dice, 且只在 **布面区域** 内计算 (背景/辊筒不参与),
  避免网络把布边外的深色背景学成 "异常".
* 布面掩码: BCE + Dice + 布边带 BCE, 显式监督前景/背景与布边位置.
* 疵点分类: 原型余弦分类 + 标签平滑 CE, 只在标注框内部监督 (弱监督).
* 度量学习: SupCon, 让同类疵点区域在嵌入空间聚拢, 为区域级分类和新类发现服务.
* 记忆库紧凑度: 正常 patch 嵌入向正常记忆库靠拢.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import move_to


# --------------------------------------------------------------------------- #
# 基础损失
# --------------------------------------------------------------------------- #
def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    if weight is not None:
        loss = loss * weight
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * targets).sum(dims)
    union = prob.sum(dims) + targets.sum(dims)
    return (1.0 - (2 * inter + eps) / (union + eps)).mean()


def refine_boxes_by_distance(
    box_mask: torch.Tensor, distance: torch.Tensor, ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """用**记忆库距离**在框内挑出异常核心, 作为更精确的像素级正样本.

    背景: 数据集只有 YOLO 框, ``boxes_to_mask`` 得到的是整块矩形. 把矩形当像素真值
    会让异常头学成"把整个框刷成异常": 边界糊、定位差, 而且框内的真实正常像素被当成
    负样本压制, 训练目标和推理目标不一致. 实测这是 pixelAUROC 卡在 0.6~0.7 的主因.

    这里用**无监督**的记忆库距离 (只由正常图建立, 与监督标签独立) 在框内保留距离最大
    的 ``ratio`` 比例像素作为正样本, 框内其余像素标记为**忽略** (既不推高也不压低).

    返回 ``(pos_mask, ignore_mask)``, 形状同 ``box_mask``.
    """
    ignore = torch.zeros_like(box_mask, dtype=torch.bool)
    if ratio <= 0.0 or ratio >= 1.0:
        return box_mask, ignore
    d = distance.detach()
    if d.shape[-2:] != box_mask.shape[-2:]:
        d = F.interpolate(d, size=box_mask.shape[-2:], mode="bilinear", align_corners=False)
    pos = torch.zeros_like(box_mask)
    inside = box_mask > 0.5
    for b in range(box_mask.shape[0]):
        sel = inside[b, 0]
        if not bool(sel.any()):
            continue
        vals = d[b, 0][sel]
        k = min(vals.numel(), max(1, int(round(vals.numel() * ratio))))
        thr = torch.topk(vals, k).values.min()
        keep = sel & (d[b, 0] >= thr)
        pos[b, 0] = keep.float()
        ignore[b, 0] = sel & (~keep)
    return pos, ignore


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在 mask>0.5 的区域计算 BCE."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weight = mask.expand_as(loss)
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


class SupConLoss(nn.Module):
    """监督对比损失 (Khosla et al.), 支持 ignore 标签."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.shape[0] < 2:
            return features.new_zeros(())
        valid = labels >= 0
        features, labels = features[valid], labels[valid]
        if features.shape[0] < 2 or labels.unique().numel() < 2:
            return features.new_zeros(()) if features.numel() else torch.tensor(0.0, device=labels.device)

        sim = features @ features.t() / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim)
        n = features.shape[0]
        self_mask = torch.eye(n, dtype=torch.bool, device=features.device)
        pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
        log_prob = sim - torch.log(exp_sim.masked_fill(self_mask, 0).sum(dim=1, keepdim=True) + 1e-8)
        pos_count = pos_mask.sum(dim=1)
        valid_rows = pos_count > 0
        if not valid_rows.any():
            return features.new_zeros(())
        mean_log_prob = (log_prob * pos_mask).sum(dim=1)[valid_rows] / pos_count[valid_rows]
        return -mean_log_prob.mean()


# --------------------------------------------------------------------------- #
# 目标构造
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_label_map(
    boxes: Sequence[torch.Tensor],
    size: tuple[int, int],
    ignore_index: int = -100,
    shrink: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """把检测框转成特征尺度上的逐像素类别图.

    ``boxes[b]`` 的每一行为 ``[cls, x1, y1, x2, y2]``, 坐标为 **相对输入图像
    的归一化坐标 (0~1)**. ``shrink`` < 1 时只取框中心区域作为正样本, 降低框
    边缘的标签噪声 (尤其适用于经纬线类细长疵点).
    """
    h, w = size
    # 必须建在 logits 所在的设备上: 否则 CUDA 训练时 cross_entropy 会报
    # "Expected all tensors to be on the same device"
    labels = torch.full((len(boxes), h, w), ignore_index, dtype=torch.long, device=device)
    for b, box in enumerate(boxes):
        if box.numel() == 0:
            continue
        box = box.to(labels.device)
        for row in box:
            cls = int(row[0].item())
            x1, y1, x2, y2 = [float(v) for v in row[1:5]]
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1) * shrink, (y2 - y1) * shrink
            x1, x2 = max(0.0, cx - bw / 2), min(float(w), cx + bw / 2)
            y1, y2 = max(0.0, cy - bh / 2), min(float(h), cy + bh / 2)
            ix1, ix2 = int(round(x1)), int(round(x2))
            iy1, iy2 = int(round(y1)), int(round(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                ix1, iy1 = int(round(cx)), int(round(cy))
                ix2, iy2 = min(w, ix1 + 1), min(h, iy1 + 1)
            labels[b, iy1:iy2, ix1:ix2] = cls
    return labels


@torch.no_grad()
def boundary_from_mask(mask: torch.Tensor, band: int = 2) -> torch.Tensor:
    """由布面掩码得到布边带 (canny-free 版: 膨胀 - 腐蚀)."""
    k = max(1, band * 2 + 1)
    dilate = F.max_pool2d(mask, k, stride=1, padding=k // 2)
    erode = -F.max_pool2d(-mask, k, stride=1, padding=k // 2)
    return (dilate - erode).clamp(0, 1)


# --------------------------------------------------------------------------- #
# 组合损失
# --------------------------------------------------------------------------- #
class FabricCriterion(nn.Module):
    """把各分支损失打包成一个模块, 便于日志与消融."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        loss_cfg = cfg.get("model", {}).get("loss", {}) if "model" in cfg else cfg.get("loss", {})
        self.cfg = loss_cfg
        ano = loss_cfg.get("anomaly", {})
        self.ano_focal = float(ano.get("focal_weight", 1.0))
        self.ano_dice = float(ano.get("dice_weight", 1.0))
        self.ano_gamma = float(ano.get("gamma", 2.0))
        self.ano_alpha = float(ano.get("alpha", 0.25))

        fab = loss_cfg.get("fabric", {})
        self.fab_bce = float(fab.get("bce_weight", 0.5))
        self.fab_dice = float(fab.get("dice_weight", 0.5))
        self.fab_edge = float(fab.get("edge_weight", 0.5))

        cls = loss_cfg.get("cls", {})
        self.cls_weight = float(cls.get("weight", 1.0))
        self.cls_smoothing = float(cls.get("label_smoothing", 0.05))
        self.cls_shrink = float(cls.get("box_shrink", 0.9))

        sup = loss_cfg.get("supcon", {})
        self.supcon_weight = float(sup.get("weight", 0.5))
        self.supcon = SupConLoss(temperature=float(sup.get("temperature", 0.1)))

        self.compact_weight = float(loss_cfg.get("compact", {}).get("weight", 0.1))
        # 框内伪标签修正: 用记忆库距离把矩形标签收缩成异常核心 (0 或 >=1 = 关闭)
        self.ano_pseudo = float(loss_cfg.get("anomaly", {}).get("pseudo_ratio", 0.0))
        self.ignore_index = -100

    # ------------------------------------------------------------------ #
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        memory=None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        device = outputs["anomaly_logit"].device
        # 目标可能停在 CPU 上 (例如自定义 collate / 单测手工喂数据): 统一搬到输出设备,
        # 避免 "Expected all tensors to be on the same device" 这类只在 GPU 上出现的报错
        targets = move_to(dict(targets), device)
        logs: Dict[str, float] = {}
        total = torch.zeros((), device=device)

        # ---------- 1. 异常分割 ----------
        ano_logit = outputs["anomaly_logit"]
        h, w = ano_logit.shape[-2:]
        defect_mask = _resize_mask(targets["defect_mask"], (h, w), mode="max")
        cloth_mask = _resize_mask(targets["cloth_mask"], (h, w), mode="area")
        region = (cloth_mask > 0.5).float()
        if region.sum() < 1:
            region = torch.ones_like(cloth_mask)

        ano_pos, ignore = defect_mask, None
        memory_distance = outputs.get("memory_distance")
        if 0.0 < self.ano_pseudo < 1.0 and memory_distance is not None:
            ano_pos, ignore = refine_boxes_by_distance(defect_mask, memory_distance, self.ano_pseudo)

        focal = sigmoid_focal_loss(
            ano_logit,
            ano_pos,
            alpha=self.ano_alpha,
            gamma=self.ano_gamma,
            reduction="none",
        )
        weight = region if ignore is None else region * (~ignore).float()
        focal = (focal * weight).sum() / weight.sum().clamp_min(1.0)
        dice = dice_loss(ano_logit * region, ano_pos * region)
        loss_ano = self.ano_focal * focal + self.ano_dice * dice
        total = total + loss_ano
        logs.update(
            anomaly=focal.item(),
            anomaly_dice=dice.item(),
            loss_ano=loss_ano.item(),
            ano_pos_ratio=float((ano_pos * region).sum() / region.sum().clamp_min(1.0)),
        )

        # ---------- 2. 布面 / 布边 ----------
        if "cloth_logit" in outputs and outputs["cloth_logit"] is not None:
            ch, cw = outputs["cloth_logit"].shape[-2:]
            cloth_gt = _resize_mask(targets["cloth_mask"], (ch, cw), mode="area")
            cloth_logit = outputs["cloth_logit"]
            loss_cloth = F.binary_cross_entropy_with_logits(cloth_logit, cloth_gt) + dice_loss(cloth_logit, cloth_gt)
            boundary = targets.get("boundary_mask")
            if boundary is None:  # 未提供真实布边时由布面掩码退化得到
                boundary = boundary_from_mask(targets["cloth_mask"], band=max(1, min(ch, cw) // 64))
            edge_gt = _resize_mask(boundary, (ch, cw), mode="max")
            pos_weight = torch.tensor([max(1.0, float((edge_gt < 0.5).sum()) / max(float((edge_gt > 0.5).sum()), 1.0))], device=device)
            loss_edge = F.binary_cross_entropy_with_logits(
                outputs["boundary_logit"], edge_gt, pos_weight=pos_weight
            )
            loss_fab = self.fab_bce * loss_cloth + self.fab_edge * loss_edge
            total = total + loss_fab
            logs.update(cloth=loss_cloth.item(), edge=loss_edge.item(), loss_fabric=loss_fab.item())

        # ---------- 3. 疵点分类 (弱监督, 仅框内) ----------
        class_logits = outputs["class_logits"]  # (B,C,h,w) 可能与异常图不同尺度
        cls_hw = tuple(class_logits.shape[-2:])
        label_map = build_label_map(
            targets["boxes"], cls_hw, self.ignore_index, self.cls_shrink, device=device
        )
        loss_cls, acc = masked_ce(class_logits, label_map, self.ignore_index, self.cls_smoothing)
        if loss_cls is not None:
            total = total + self.cls_weight * loss_cls
            logs.update(cls=loss_cls.item(), cls_acc=acc)

        # ---------- 4. 区域嵌入的 SupCon ----------
        if self.supcon_weight > 0:
            region_emb, region_labels = pooled_region_embeddings(
                outputs["embedding"],
                targets["boxes"],
                tuple(outputs["embedding"].shape[-2:]),
            )
            loss_sc = self.supcon(region_emb, region_labels)
            if loss_sc.requires_grad:
                total = total + self.supcon_weight * loss_sc
            logs.update(supcon=float(loss_sc.detach()))

        # ---------- 5. 正常嵌入紧凑度 ----------
        if memory is not None and memory.is_ready and self.compact_weight > 0:
            from .models.memory import compact_loss

            emb = outputs["embedding"]
            region_emb = _resize_mask(region, tuple(emb.shape[-2:]), mode="area")
            loss_c = compact_loss(memory, emb * region_emb)
            total = total + self.compact_weight * loss_c
            logs.update(compact=float(loss_c.detach()))

        logs["loss_total"] = float(total.detach())
        return total, logs


def _resize_mask(mask: torch.Tensor, size: tuple[int, int], mode: str = "area") -> torch.Tensor:
    mask = mask.float()
    if mask.shape[-2:] == size:
        return mask
    if mode == "max":
        return F.adaptive_max_pool2d(mask, size)
    return F.interpolate(mask, size=size, mode="area")


def masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    smoothing: float = 0.0,
) -> tuple[Optional[torch.Tensor], float]:
    """仅在有效标签处计算 CE, 返回 (loss, acc)."""
    labels = labels.to(logits.device)  # 防御: 调用方漏搬设备时也只在这里修正
    valid = labels != ignore_index
    if not valid.any():
        return None, 0.0
    logits_flat = logits.permute(0, 2, 3, 1)[valid]  # (N,C)
    labels_flat = labels[valid]
    loss = F.cross_entropy(logits_flat, labels_flat, label_smoothing=smoothing)
    with torch.no_grad():
        acc = (logits_flat.argmax(dim=1) == labels_flat).float().mean().item()
    return loss, acc


@torch.no_grad()
def pooled_region_embeddings(
    embedding: torch.Tensor,
    boxes: Sequence[torch.Tensor],
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """按标注框对嵌入做掩码平均池化, 用于 SupCon / 新类发现."""
    h, w = size
    b, d = embedding.shape[:2]
    feats: List[torch.Tensor] = []
    labels: List[int] = []
    for i in range(b):
        box = boxes[i]
        if box is None or len(box) == 0:
            continue
        box = box.to(embedding.device)
        for row in box:
            cls = int(row[0].item())
            x1, y1, x2, y2 = [float(v) for v in row[1:5]]
            ix1, ix2 = int(x1 * w), int(round(x2 * w))
            iy1, iy2 = int(y1 * h), int(round(y2 * h))
            ix1, iy1 = max(0, min(ix1, w - 1)), max(0, min(iy1, h - 1))
            ix2, iy2 = max(ix1 + 1, min(ix2, w)), max(iy1 + 1, min(iy2, h))
            patch = embedding[i, :, iy1:iy2, ix1:ix2]
            feats.append(F.normalize(patch.mean(dim=(-2, -1)), dim=0))
            labels.append(cls)
    if not feats:
        return embedding.new_zeros((0, d)), torch.zeros(0, dtype=torch.long, device=embedding.device)
    return torch.stack(feats), torch.tensor(labels, device=embedding.device)
