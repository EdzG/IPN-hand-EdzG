import torch
from torch import nn

from .models import resnet, resnext, resnetl, c3d, mobilenetv2, shufflenetv2


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnetl', 'resnext', 'c3d', 'mobilenetv2', 'shufflenetv2'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 50]
        from .models.resnet import get_fine_tuning_parameters
        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.model == 'resnetl':
        assert opt.model_depth in [10]
        from .models.resnetl import get_fine_tuning_parameters
        model = resnetl.resnetl10(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'resnext':
        assert opt.model_depth in [101]
        from .models.resnext import get_fine_tuning_parameters
        model = resnext.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'c3d':
        assert opt.model_depth in [10]
        from .models.c3d import get_fine_tuning_parameters
        model = c3d.c3d_v1(
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            num_classes=opt.n_classes)

    elif opt.model == 'mobilenetv2':
        from .models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.mob_v2(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)

    elif opt.model == 'shufflenetv2':
        from .models.shufflenetv2 import get_fine_tuning_parameters
        model = shufflenetv2.shf_v2(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)

    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    # --- Load pretrained weights ---
    # All weight surgery is done on the bare (non-DataParallel) model so that
    # state_dict key names are consistent regardless of CUDA availability.
    if opt.pretrain_path:
        print(f'loading pretrained model {opt.pretrain_path}')
        pretrain = torch.load(opt.pretrain_path, map_location='cpu', weights_only=False)
        assert opt.arch == pretrain['arch']

        # Checkpoints are saved after DataParallel wrapping, so keys are
        # prefixed with 'module.'.  Strip that prefix to match the bare model.
        state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in pretrain['state_dict'].items()
        }

        if opt.pretrain_dataset == 'jester':
            if opt.sample_duration < 32 and opt.model != 'c3d':
                model = _modify_first_conv_layer(model, 3, 3)
            if opt.model in ['mobilenetv2', 'shufflenetv2']:
                state_dict.pop('classifier.1.weight', None)
                state_dict.pop('classifier.1.bias', None)
            else:
                state_dict.pop('fc.weight', None)
                state_dict.pop('fc.bias', None)
            model.load_state_dict(state_dict, strict=False)

        elif opt.pretrain_dataset == opt.dataset:
            model.load_state_dict(state_dict)

        elif opt.pretrain_dataset in ['egogesture', 'nv', 'denso']:
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            model.load_state_dict(state_dict, strict=False)

    # --- Modality adaptation (RGB only) ---
    if opt.model != 'c3d' and opt.dataset != 'jester' and not opt.no_first_lay:
        model = _modify_first_conv_layer(model, 3, 3)

    # Check that the first conv temporal kernel fits the clip length
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 range(len(modules))))[0]
    if modules[first_conv_idx].kernel_size[0] > opt.sample_duration:
        model = _modify_first_conv_layer(model, int(opt.sample_duration / 2), 1)

    # --- Replace classification head ---
    if opt.model == 'c3d':
        model.fc = nn.Linear(model.fc[0].in_features, model.fc[0].out_features)
    elif opt.model in ['mobilenetv2', 'shufflenetv2']:
        model.classifier = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(model.classifier[1].in_features, opt.n_finetune_classes))
    else:
        model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)

    # --- Wrap in DataParallel (CUDA only), then move to device ---
    if not opt.no_cuda:
        model = nn.DataParallel(model, device_ids=None)

    model = model.to(device)
    parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
    return model, parameters

def _modify_first_conv_layer(base_model, new_kernel_size1, new_filter_num):
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
 
    new_conv = nn.Conv3d(new_filter_num, conv_layer.out_channels, kernel_size=(new_kernel_size1,7,7),
                         stride=(1,2,2), padding=(1,3,3), bias=False)
    layer_name = list(container.state_dict().keys())[0][:-7]

    setattr(container, layer_name, new_conv)
    return base_model


