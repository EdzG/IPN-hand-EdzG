from .datasets.jester import Jester
from .datasets.ipn import IPN
from .datasets.ipn_online import IPNOnline


def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['jester', 'ipn']

    subset = ['training', 'validation'] if opt.train_validate else 'training'

    if opt.dataset == 'jester':
        return Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)
    elif opt.dataset == 'ipn':
        return IPN(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['jester', 'ipn']

    if opt.dataset == 'jester':
        return Jester(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ipn':
        return IPN(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['jester', 'ipn']
    assert opt.test_subset in ['val', 'test']

    subset = 'validation' if opt.test_subset == 'val' else 'testing'

    if opt.dataset == 'jester':
        return Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            modality=opt.modality,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ipn':
        return IPN(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            modality=opt.modality)


def get_online_data(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset == 'ipn'

    return IPNOnline(
        opt.annotation_path,
        opt.video_path,
        opt.whole_path,
        opt.n_val_samples,
        spatial_transform,
        temporal_transform,
        target_transform,
        modality=opt.modality,
        stride_len=opt.stride_len,
        sample_duration=opt.sample_duration)
