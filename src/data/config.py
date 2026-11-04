"""读取 dataset/fabric19.yaml (YOLO 格式标注的 19 类布匹缺陷数据集) 并解析路径.

yaml 字段:
  path / train / val / normaltrain / normalval / nc / names
其中 train,val 为带 YOLO 标签的图片目录 (images/train, images/val),
normaltrain / normalval 为纯正常样本目录 (无标签, 用于无监督异常检测训练与校准),
labels 目录与 images 目录同级 (labels/train, labels/val).

路径解析: 绝对路径且存在则直接用; 否则尝试相对 yaml 所在目录 / 相对 path / 相对 cwd;
也可用环境变量 FABRIC19_ROOT 指向数据集根目录 (path 字段).
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

UNKNOWN_CLASS_NAME = '未知类型'

DEFAULT_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'dataset', 'fabric19.yaml')


def _first_existing(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


@dataclass
class FabricConfig:
    yaml_path: str
    root: Optional[str] = None
    train_images: Optional[str] = None
    val_images: Optional[str] = None
    normaltrain_images: Optional[str] = None
    normalval_images: Optional[str] = None
    train_labels: Optional[str] = None
    val_labels: Optional[str] = None
    nc: int = 19
    names: List[str] = field(default_factory=list)

    @property
    def class_names(self) -> List[str]:
        """19 类缺陷 + 追加的 '未知类型' (索引 = nc)."""
        return list(self.names) + [UNKNOWN_CLASS_NAME]

    @property
    def unknown_id(self) -> int:
        return self.nc

    def summary(self) -> str:
        return (f'fabric19: nc={self.nc}, 类别={self.names}\n'
                f'  train_images     = {self.train_images}\n'
                f'  train_labels     = {self.train_labels}\n'
                f'  val_images       = {self.val_images}\n'
                f'  val_labels       = {self.val_labels}\n'
                f'  normaltrain      = {self.normaltrain_images}\n'
                f'  normalval        = {self.normalval_images}')


def _resolve(raw, yaml_dir, root_candidates):
    if raw is None:
        return None
    raw = str(raw)
    cands = [raw]
    for base in [yaml_dir] + root_candidates:
        if base:
            cands.append(os.path.join(base, raw))
            cands.append(os.path.join(base, os.path.basename(raw)))
    found = _first_existing(cands)
    return found if found else os.path.abspath(raw if os.path.isabs(raw) else os.path.join(yaml_dir, raw))


def _labels_dir_for(images_dir: Optional[str]) -> Optional[str]:
    """images/train -> ../labels/train (与 yaml 注释中的目录结构一致)."""
    if not images_dir:
        return None
    parent = os.path.dirname(images_dir)          # .../images
    grand = os.path.dirname(parent)                # dataset root
    cand = os.path.join(grand, 'labels', os.path.basename(images_dir))
    return cand if os.path.isdir(cand) else os.path.join(parent, 'labels', os.path.basename(images_dir))


def load_fabric_config(yaml_path: str = DEFAULT_YAML, root_override: Optional[str] = None) -> FabricConfig:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'数据集配置不存在: {yaml_path}')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    root_raw = root_override or os.environ.get('FABRIC19_ROOT') or y.get('path')
    root_candidates = []
    if root_raw:
        root_candidates.append(str(root_raw))
        root_candidates.append(os.path.join(yaml_dir, str(root_raw)))

    cfg = FabricConfig(yaml_path=os.path.abspath(yaml_path))
    cfg.root = _first_existing(root_candidates)
    cfg.train_images = _resolve(y.get('train'), yaml_dir, root_candidates)
    cfg.val_images = _resolve(y.get('val'), yaml_dir, root_candidates)
    cfg.normaltrain_images = _resolve(y.get('normaltrain'), yaml_dir, root_candidates)
    cfg.normalval_images = _resolve(y.get('normalval'), yaml_dir, root_candidates)
    cfg.nc = int(y.get('nc', 19))
    cfg.names = [str(n) for n in y.get('names', [])]
    cfg.train_labels = _resolve(y.get('train_labels'), yaml_dir, root_candidates) or _labels_dir_for(cfg.train_images)
    cfg.val_labels = _resolve(y.get('val_labels'), yaml_dir, root_candidates) or _labels_dir_for(cfg.val_images)
    return cfg
