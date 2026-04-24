from pathlib import Path

import torch

from .mean import get_mean, get_std
from .transforms.spatial_transforms import CenterCrop, Compose, Normalize, Scale, ToTensor


def resolve_dataset_paths(opt) -> None:
    """Mutates opt: sets video_path and annotation_path from the dataset root."""
    match opt.dataset:
        case 'ipn':
            root = Path(opt.ipn_root_path)
            opt.video_path = str(root / opt.ipn_video_path)
            opt.annotation_path = str(root / opt.ipn_annotation_path)
        case 'jester':
            root = Path(opt.jester_root_path)
            opt.video_path = str(root / opt.jester_video_path)
            opt.annotation_path = str(root / opt.jester_annotation_path)


def setup_opt(opt) -> None:
    """Mutates opt: computes scales, arch, store_name, mean, std, and seeds the RNG."""
    opt.scales = [opt.initial_scale * (opt.scale_step ** i) for i in range(opt.n_scales)]
    opt.arch = f'{opt.model}-{opt.model_depth}'
    opt.store_name = f'{opt.store_name}_{opt.arch}'
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)
    torch.manual_seed(opt.manual_seed)


def build_norm_method(opt) -> Normalize:
    """Returns the Normalize transform for the configured normalization mode."""
    if opt.no_mean_norm and not opt.std_norm:
        return Normalize([0, 0, 0], [1, 1, 1])
    if not opt.std_norm:
        return Normalize(opt.mean, [1, 1, 1])
    return Normalize(opt.mean, opt.std)


def build_inference_transform(opt, norm_method: Normalize) -> Compose:
    """Standard center-crop spatial transform for inference/validation."""
    return Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value),
        norm_method,
    ])


def load_checkpoint(
    model,
    resume_path: str,
    arch: str,
    optimizer=None,
    fine_tuning: bool = False,
) -> int:
    """Loads a checkpoint with arch validation. Returns begin_epoch."""
    print(f'Loading checkpoint: {resume_path}', flush=True)
    checkpoint = torch.load(resume_path, weights_only=False)
    assert arch == checkpoint['arch'], (
        f'Architecture mismatch: expected {arch}, got {checkpoint["arch"]}'
    )
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return 1 if fine_tuning else checkpoint['epoch']
