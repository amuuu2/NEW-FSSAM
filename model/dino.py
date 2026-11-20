"""Load DINOv2 source and weights locally, without accessing Torch Hub online."""
from pathlib import Path

import torch


def load_dinov2(version):
    root = Path(__file__).resolve().parents[1]
    source_path = root / 'third_party' / 'dinov2'
    weight_path = root / 'pretrained' / f'{version}_pretrain.pth'
    if not (source_path / 'hubconf.py').is_file():
        raise FileNotFoundError(f'Local DINOv2 source not found: {source_path}')
    if not weight_path.is_file():
        raise FileNotFoundError(f'Local DINOv2 checkpoint not found: {weight_path}')
    model = torch.hub.load(
        str(source_path), version, source='local', pretrained=False,
    )
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model
