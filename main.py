import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from src.opts import parse_opts_offline
from src.model import generate_model
from src.setup import resolve_dataset_paths, setup_opt, build_norm_method, build_inference_transform, load_checkpoint
from src.transforms.spatial_transforms import (
    Compose, ToTensor,
    MultiScaleRandomCrop, MultiScaleCornerCrop, SpatialElasticDisplacement,
)
from src.transforms.temporal_transforms import (
    TemporalRandomCrop, TemporalPadRandomCrop, TemporalBeginCrop, TemporalCenterCrop,
)
from src.transforms.target_transforms import ClassLabel, VideoID
from src.dataset import get_training_set, get_validation_set, get_test_set
from src.utils import Logger
from src.train import train_epoch
from src.validation import val_epoch, val_epoch_true
import src.test as test


def save_checkpoint(state: dict, is_best: bool, opt: Namespace) -> None:
    result = Path(opt.result_path)
    checkpoint_path = result / f'{opt.store_name}_checkpoint.pth'
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, result / f'{opt.store_name}_best.pth')


def adjust_learning_rate(optimizer: optim.Optimizer, epoch: int, opt: Namespace) -> None:
    lr = opt.learning_rate * (0.1 ** sum(epoch >= np.array(opt.lr_steps)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main() -> None:
    opt = parse_opts_offline()
    resolve_dataset_paths(opt)
    setup_opt(opt)
    print(opt, flush=True)

    result_path = Path(opt.result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    with open(result_path / f'opts_{opt.store_name}.json', 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    model, parameters = generate_model(opt)
    print(model, flush=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params:,}', flush=True)

    if opt.weighted:
        print('Weighted loss enabled', flush=True)
        weight = (
            torch.tensor([1.0, 3.0]) if opt.n_finetune_classes == 2
            else torch.ones(opt.n_finetune_classes)
        )
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(weight=weight)
    if not opt.no_cuda:
        criterion = criterion.cuda()

    norm_method = build_norm_method(opt)

    if not opt.no_train:
        assert opt.train_crop in {'random', 'corner', 'center'}
        match opt.train_crop:
            case 'random':
                crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
            case 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            case 'center':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c'])

        match opt.train_temporal:
            case 'random':
                temp_method = TemporalRandomCrop(opt.sample_duration)
            case 'ranpad':
                temp_method = TemporalPadRandomCrop(opt.sample_duration, opt.temporal_pad)
            case _:
                raise ValueError(f'Unknown train_temporal: {opt.train_temporal!r}')

        spatial_transform = Compose([
            crop_method,
            SpatialElasticDisplacement(),
            ToTensor(opt.norm_value),
            norm_method,
        ])
        temporal_transform = Compose([temp_method])
        training_data = get_training_set(opt, spatial_transform, temporal_transform, ClassLabel())
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True,
            persistent_workers=opt.n_threads > 0,
        )
        train_logger = Logger(
            str(result_path / f'train_{opt.store_name}.log'),
            ['epoch', 'loss', 'acc', 'precision', 'recall', 'lr'],
        )
        train_batch_logger = Logger(
            str(result_path / f'train_batch_{opt.store_name}.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'precision', 'recall', 'lr'],
        )

        dampening = 0 if opt.nesterov else opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov,
        )

    if not opt.no_val:
        spatial_transform = build_inference_transform(opt, norm_method)
        if opt.true_valid:
            val_batch = 1
            temporal_transform = Compose([TemporalBeginCrop(opt.sample_duration)])
        else:
            val_batch = opt.batch_size
            temporal_transform = Compose([TemporalCenterCrop(opt.sample_duration)])

        validation_data = get_validation_set(opt, spatial_transform, temporal_transform, ClassLabel())
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=val_batch,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            persistent_workers=opt.n_threads > 0,
        )
        val_logger = Logger(
            str(result_path / f'val_{opt.store_name}.log'),
            ['epoch', 'loss', 'acc', 'precision', 'recall'],
        )

    if opt.resume_path:
        opt.begin_epoch = load_checkpoint(
            model, opt.resume_path, opt.arch,
            optimizer=optimizer if not opt.no_train else None,
            fine_tuning=opt.fine_tuning,
        )

    best_prec1 = 0.0
    ep_best = 0
    print('Training started', flush=True)

    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            adjust_learning_rate(optimizer, epoch, opt)
            train_epoch(epoch, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)

        if not opt.no_val:
            if opt.true_valid:
                validation_loss, prec1 = val_epoch_true(epoch, val_loader, model, criterion, opt, val_logger)
            else:
                validation_loss, prec1 = val_epoch(epoch, val_loader, model, criterion, opt, val_logger)

            is_best = prec1 > best_prec1
            if is_best:
                ep_best = epoch
            best_prec1 = max(prec1, best_prec1)
            print(f'  Val acc: {prec1:.4f}  (best: {best_prec1:.4f} @ ep {ep_best})', flush=True)

            save_checkpoint({
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, opt)

    if opt.test:
        spatial_transform = build_inference_transform(opt, norm_method)
        temporal_transform = Compose([TemporalCenterCrop(opt.sample_duration)])
        test_data = get_test_set(opt, spatial_transform, temporal_transform, VideoID())
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            persistent_workers=opt.n_threads > 0,
        )
        test.test(test_loader, model, opt, test_data.class_names)


if __name__ == '__main__':
    main()
