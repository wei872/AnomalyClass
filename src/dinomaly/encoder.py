"""按名字加载冻结 encoder (backbone).

支持:
  dinov2reg_vit_small_14 / dinov2reg_vit_base_14 / dinov2reg_vit_large_14  (推荐, 官方 reg4 权重)
  dinov2_vit_small_14 / dinov2_vit_base_14 / dinov2_vit_large_14
  tiny_test_14  (本项目冒烟测试用, 随机初始化, 无需下载权重)

权重默认下载到 ./backbones/weights (可通过 --weights-dir 修改).
下载失败 (如离线环境) 时: 若 allow_random_init=True 则退化为随机初始化 (仅用于代码冒烟测试).
"""
import os

from . import vit_dinov2

_DINOV2_URLS = {
    # arch: (url, num_register_tokens)
    ('small', True): ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth', 4),
    ('base', True): ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth', 4),
    ('large', True): ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth', 4),
    ('small', False): ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', 0),
    ('base', False): ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth', 0),
    ('large', False): ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 0),
}

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'backbones', 'weights')


def parse_backbone_name(name: str):
    """'dinov2reg_vit_small_14' -> (family='dinov2', reg=True, arch='small', patch=14)."""
    name = name.lower()
    if name == 'tiny_test_14':
        return 'tiny', False, 'tiny', 14
    parts = name.split('_')
    arch, patch = parts[-2], int(parts[-1])
    family = parts[0]
    if family == 'dinov2reg':
        return 'dinov2', True, arch, patch
    if family == 'dinov2':
        return 'dinov2', False, arch, patch
    raise ValueError(f'不支持的 backbone: {name} (支持 dinov2reg_vit_* / dinov2_vit_* / tiny_test_14)')


def load_encoder(name: str, weights_dir: str = None, allow_random_init: bool = False, logger=None):
    """构建并加载预训练 encoder, 返回 nn.Module (含 prepare_tokens / blocks / num_register_tokens)."""
    log = logger or (lambda *a, **k: print(*a))
    family, reg, arch, patch = parse_backbone_name(name)
    weights_dir = weights_dir or DEFAULT_WEIGHTS_DIR
    os.makedirs(weights_dir, exist_ok=True)

    if family == 'tiny':
        model = vit_dinov2.build_vit('tiny', patch_size=patch, num_register_tokens=0, img_size=224)
        model.num_register_tokens = 0
        log('[encoder] tiny_test_14: 随机初始化 (仅冒烟测试)')
        return model

    n_reg = 4 if reg else 0
    model = vit_dinov2.build_vit(arch, patch_size=patch, num_register_tokens=n_reg, img_size=518)
    url, _ = _DINOV2_URLS[(arch, reg)]
    ckpt_path = os.path.join(weights_dir, os.path.basename(url))
    if not os.path.exists(ckpt_path):
        log(f'[encoder] 下载预训练权重: {url}')
        try:
            from torch.hub import download_url_to_file
            download_url_to_file(url, ckpt_path, progress=True)
        except Exception as e:
            if allow_random_init:
                log(f'[encoder] 警告: 权重下载失败 ({e}), 使用随机初始化 (仅冒烟测试!)')
                model.num_register_tokens = n_reg
                return model
            raise RuntimeError(
                f'下载 DINOv2 权重失败: {e}\n'
                f'请手动下载 {url} 放到 {ckpt_path}, 或检查网络后重试.')
    log(f'[encoder] 加载权重: {ckpt_path}')
    from ..util import torch_load
    state = torch_load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    missing, unexpected = model.load_state_dict(state, strict=False)
    # pos_embed 尺寸不匹配时由 interpolate_pos_encoding 处理, 允许缺失
    missing = [k for k in missing if k != 'pos_embed']
    if missing or unexpected:
        log(f'[encoder] 权重加载提示: missing={missing}, unexpected={unexpected}')
    model.num_register_tokens = n_reg
    return model


def encoder_embed_dim(name: str) -> int:
    _, _, arch, _ = parse_backbone_name(name)
    return {'small': 384, 'base': 768, 'large': 1024, 'tiny': 64}[arch]


def encoder_num_heads(name: str) -> int:
    _, _, arch, _ = parse_backbone_name(name)
    return {'small': 6, 'base': 12, 'large': 16, 'tiny': 2}[arch]
