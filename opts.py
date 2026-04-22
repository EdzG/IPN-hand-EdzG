import argparse


def parse_opts_offline():
    # Offline = not real-time. Single model, no separate det/clf architecture.
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------------------
    # Mode
    # pretrain    : train on Jester from scratch (intended for later transfer)
    # finetune    : load a Jester checkpoint and fine-tune on IPN
    # scratch     : train from scratch on whichever dataset --dataset specifies
    # -------------------------------------------------------------------------
    parser.add_argument('--mode', default='pretrain', type=str,
        choices=['pretrain', 'finetune', 'scratch'],
        help='Training mode: pretrain=Jester from scratch, finetune=Jester->IPN, scratch=either dataset from scratch')

    # -------------------------------------------------------------------------
    # Dataset selection (used by scratch mode and for mean/annotation routing)
    # -------------------------------------------------------------------------
    parser.add_argument('--dataset', default='jester', type=str,
        choices=['ipn', 'jester'],
        help='Dataset to use (ipn | jester). In pretrain mode this is always jester. '
             'In finetune mode this is always ipn. In scratch mode set this explicitly.')

    # -------------------------------------------------------------------------
    # Paths — kept separate so you never have to edit paths between runs,
    # just change --mode
    # -------------------------------------------------------------------------
    parser.add_argument('--jester_root_path', default='/root/data/Jester', type=str,
        help='Root directory of Jester data')
    parser.add_argument('--jester_video_path', default='jpg', type=str,
        help='Video frames directory relative to jester_root_path')
    parser.add_argument('--jester_annotation_path', default='jester.json', type=str,
        help='Annotation file for Jester')

    parser.add_argument('--ipn_root_path', default='/root/data/IPN', type=str,
        help='Root directory of IPN-Hand data')
    parser.add_argument('--ipn_video_path', default='jpg', type=str,
        help='Video frames directory relative to ipn_root_path')
    parser.add_argument('--ipn_annotation_path', default='ipn.json', type=str,
        help='Annotation file for IPN-Hand')

    parser.add_argument('--result_path', default='results', type=str,
        help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str,
        help='Name to store checkpoints')

    # -------------------------------------------------------------------------
    # Class counts
    # n_classes          = classes in the pretrained/source model (Jester: 27)
    # n_finetune_classes = classes in the new head after replacement (IPN: 13)
    # pretrain  : both should be 27
    # finetune  : n_classes=27 (matches Jester checkpoint), n_finetune=13
    # scratch   : set both to match your chosen dataset (27 or 13)
    # -------------------------------------------------------------------------
    parser.add_argument('--n_classes', default=27, type=int,
        help='Number of classes in the pretrained model (jester: 27, ipn: 13)')
    parser.add_argument('--n_finetune_classes', default=13, type=int,
        help='Number of classes for the fine-tuned head (ipn: 13, jester: 27)')

    # IPN-Hand has an explicit no-gesture class. Enable this to add it on top
    # of the 13 gesture classes, making the effective output 14.
    parser.add_argument('--background_class', action='store_true',
        help='Add a no-gesture background class (IPN-Hand only, makes n_finetune_classes effectively +1)')
    parser.set_defaults(background_class=False)

    # -------------------------------------------------------------------------
    # Mean normalization
    # IMPORTANT: must match the dataset being trained on.
    # pretrain/scratch-jester -> jester
    # finetune/scratch-ipn    -> ipn
    # -------------------------------------------------------------------------
    parser.add_argument('--mean_dataset', default='jester', type=str,
        choices=['ipn', 'jester'],
        help='Dataset whose channel means are used for normalization. Must match active dataset.')
    parser.add_argument('--no_mean_norm', action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)

    # -------------------------------------------------------------------------
    # Pretrained / resume weights
    # During finetune, point --pretrain_path at the Jester checkpoint.
    # -------------------------------------------------------------------------
    parser.add_argument('--pretrain_path', default='', type=str,
        help='Path to pretrained Jester model (.pth). Used in finetune mode.')
    parser.add_argument('--pretrain_dataset', default='jester', type=str,
        choices=['jester', ''],
        help='Dataset the pretrained model was trained on')
    parser.add_argument('--resume_path', default='', type=str,
        help='Path to a mid-run checkpoint to resume from (.pth)')

    # -------------------------------------------------------------------------
    # Fine-tuning control
    # ft_begin_index: which residual block to start unfreezing from.
    # 0 = unfreeze everything, 4 = only last block (recommended starting point
    # for Jester -> IPN). Lower if validation plateaus.
    # -------------------------------------------------------------------------
    parser.add_argument('--ft_begin_index', default=4, type=int,
        help='Block index to start fine-tuning from. 0=all layers, 4=last block only.')
    parser.add_argument('--fine_tuning', action='store_true',
        help='If true, fine-tuning starts from epoch 1.')
    parser.set_defaults(fine_tuning=False)

    # -------------------------------------------------------------------------
    # Input shape
    # -------------------------------------------------------------------------
    parser.add_argument('--modality', default='RGB', type=str,
        help='Modality of input data (RGB | Depth | RGB-D)')
    parser.add_argument('--sample_size', default=112, type=int,
        help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int,
        help='Temporal duration of inputs (jester: 16, ipn: 8 or 16)')

    # -------------------------------------------------------------------------
    # Augmentation / preprocessing
    # -------------------------------------------------------------------------
    parser.add_argument('--initial_scale', default=1.0, type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str,
        help='Spatial crop method in training (random | corner | center)')
    parser.add_argument('--train_temporal', default='random', type=str,
        help='Temporal transformation method in training (random | ranpad)')
    parser.add_argument('--temporal_pad', default=0, type=int,
        help='Pad number for temporal transformation method (ranpad)')
    parser.add_argument('--no_hflip', action='store_true',
        help='Disable horizontal flipping. Consider enabling for hand gestures '
             'where left/right handedness matters.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int,
        help='1: inputs in [0-255]. 255: inputs in [0-1].')
    parser.add_argument('--adap_temp', action='store_true',
        help='Adaptive temporal cropping when input frames exceed sample_duration')
    parser.set_defaults(adap_temp=False)

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # Recommended: lr=0.1 for pretrain/scratch, lr=0.001 for finetune
    # -------------------------------------------------------------------------
    parser.add_argument('--learning_rate', default=0.1, type=float,
        help='Initial learning rate. Use ~0.1 for pretrain/scratch, ~0.001 for finetune.')
    parser.add_argument('--lr_steps', default=[10, 25, 50, 80, 100], type=float,
        nargs='+', metavar='LRSteps',
        help='Epochs at which to decay learning rate by 10')
    parser.add_argument('--lr_patience', default=10, type=int,
        help='Patience of ReduceLROnPlateau scheduler')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--dampening', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--nesterov', action='store_true',
        help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str,
        help='Optimizer (sgd)')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--begin_epoch', default=1, type=int,
        help='Epoch to start training from. Loads resume_path if > 1.')
    parser.add_argument('--n_val_samples', default=3, type=int,
        help='Number of validation samples per activity')
    parser.add_argument('--checkpoint', default=10, type=int,
        help='Save a checkpoint every this many epochs')
    parser.add_argument('--weighted', action='store_true',
        help='If true, loss is class-weighted')
    parser.set_defaults(weighted=False)

    # -------------------------------------------------------------------------
    # Model architecture
    # -------------------------------------------------------------------------
    parser.add_argument('--model', default='resnet', type=str,
        help='Model type (resnet | preresnet | wideresnet | resnext | densenet)')
    parser.add_argument('--model_depth', default=18, type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int,
        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int,
        help='ResNeXt cardinality')
    parser.add_argument('--no_first_lay', action='store_true',
        help='If true, first conv layer is replaced')
    parser.set_defaults(no_first_lay=False)
    parser.add_argument('--width_mult', default=1.0, type=float,
        help='Width multiplier to scale number of filters')

    # -------------------------------------------------------------------------
    # Validation / test settings
    # -------------------------------------------------------------------------
    parser.add_argument('--no_train', action='store_true')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true')
    parser.set_defaults(no_val=False)
    parser.add_argument('--true_valid', action='store_true',
        help='If true, avg recognition per clip is used instead of center crop only')
    parser.set_defaults(true_valid=False)
    parser.add_argument('--train_validate', action='store_true',
        help='If true, validation is run during training')
    parser.set_defaults(train_validate=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--scale_in_test', default=1.0, type=float)
    parser.add_argument('--crop_position_in_test', default='c', type=str,
        help='Crop position in test (c | tl | tr | bl | br)')
    parser.add_argument('--no_softmax_in_test', action='store_true')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--clf_threshold', default=0.1, type=float,
        help='Score threshold below which output is treated as no-gesture')

    # -------------------------------------------------------------------------
    # Misc
    # -------------------------------------------------------------------------
    parser.add_argument('--no_cuda', action='store_true')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=4, type=int)
    parser.add_argument('--manual_seed', default=1, type=int)

    args = parser.parse_args()
    return args


def parse_opts_online():
    # Real-time inference with separate detector and classifier architecture.
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------------------
    # Mode
    # pretrain    : train on Jester from scratch (intended for later transfer)
    # finetune    : load a Jester checkpoint and fine-tune on IPN
    # scratch     : train from scratch on whichever dataset --dataset specifies
    # -------------------------------------------------------------------------
    parser.add_argument('--mode', default='pretrain', type=str,
        choices=['pretrain', 'finetune', 'scratch'],
        help='Training mode: pretrain=Jester from scratch, finetune=Jester->IPN, scratch=either dataset from scratch')

    # -------------------------------------------------------------------------
    # Dataset selection (used by scratch mode and for mean/annotation routing)
    # -------------------------------------------------------------------------
    parser.add_argument('--dataset', default='jester', type=str,
        choices=['ipn', 'jester'],
        help='Dataset to use (ipn | jester). In pretrain mode this is always jester. '
             'In finetune mode this is always ipn. In scratch mode set this explicitly.')

    # -------------------------------------------------------------------------
    # Paths — kept separate so you never have to edit paths between runs
    # -------------------------------------------------------------------------
    parser.add_argument('--jester_root_path', default='/root/data/Jester', type=str,
        help='Root directory of Jester data')
    parser.add_argument('--jester_video_path', default='jpg', type=str,
        help='Video frames directory relative to jester_root_path')
    parser.add_argument('--jester_annotation_path', default='jester.json', type=str,
        help='Annotation file for Jester')

    parser.add_argument('--ipn_root_path', default='/root/data/IPN', type=str,
        help='Root directory of IPN-Hand data')
    parser.add_argument('--ipn_video_path', default='jpg', type=str,
        help='Video frames directory relative to ipn_root_path')
    parser.add_argument('--ipn_annotation_path', default='ipn.json', type=str,
        help='Annotation file for IPN-Hand')

    parser.add_argument('--result_path', default='results', type=str,
        help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str,
        help='Name to store checkpoints')

    # -------------------------------------------------------------------------
    # Class counts
    # n_classes          = classes in the pretrained/source model (Jester: 27)
    # n_finetune_classes = classes in the new head after replacement (IPN: 13)
    # pretrain  : both should be 27
    # finetune  : n_classes=27 (matches Jester checkpoint), n_finetune=13
    # scratch   : set both to match your chosen dataset (27 or 13)
    # -------------------------------------------------------------------------
    parser.add_argument('--n_classes', default=27, type=int,
        help='Number of classes in the pretrained model (jester: 27, ipn: 13)')
    parser.add_argument('--n_finetune_classes', default=13, type=int,
        help='Number of classes for the fine-tuned head (ipn: 13, jester: 27)')

    parser.add_argument('--n_classes_det', default=27, type=int,
        help='Detector: class count in pretrained model')
    parser.add_argument('--n_finetune_classes_det', default=13, type=int,
        help='Detector: class count after head replacement')

    parser.add_argument('--n_classes_clf', default=27, type=int,
        help='Classifier: class count in pretrained model')
    parser.add_argument('--n_finetune_classes_clf', default=13, type=int,
        help='Classifier: class count after head replacement')

    # IPN-Hand has an explicit no-gesture class. Enable this to add it on top
    # of the 13 gesture classes, making the effective output 14.
    parser.add_argument('--background_class', action='store_true',
        help='Add a no-gesture background class (IPN-Hand only, makes n_finetune_classes effectively +1)')
    parser.set_defaults(background_class=False)

    # -------------------------------------------------------------------------
    # Mean normalization
    # IMPORTANT: must match the dataset being trained on.
    # pretrain/scratch-jester -> jester
    # finetune/scratch-ipn    -> ipn
    # -------------------------------------------------------------------------
    parser.add_argument('--mean_dataset', default='jester', type=str,
        choices=['ipn', 'jester'],
        help='Dataset whose channel means are used for normalization. Must match active dataset.')
    parser.add_argument('--no_mean_norm', action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)

    # -------------------------------------------------------------------------
    # Modality
    # -------------------------------------------------------------------------
    parser.add_argument('--modality', default='RGB', type=str,
        help='Modality of input data (RGB | Flow | RGBFlow)')
    parser.add_argument('--modality_det', default='RGB', type=str)
    parser.add_argument('--modality_clf', default='RGB', type=str)

    # -------------------------------------------------------------------------
    # Input shape
    # -------------------------------------------------------------------------
    parser.add_argument('--sample_size', default=112, type=int,
        help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int,
        help='Temporal duration of inputs (jester: 16, ipn: 8 or 16)')
    parser.add_argument('--sample_duration_det', default=16, type=int)
    parser.add_argument('--sample_duration_clf', default=16, type=int)

    # -------------------------------------------------------------------------
    # Augmentation / preprocessing
    # -------------------------------------------------------------------------
    parser.add_argument('--initial_scale', default=1.0, type=float)
    parser.add_argument('--n_scales', default=5, type=int)
    parser.add_argument('--scale_step', default=0.84089641525, type=float)
    parser.add_argument('--train_crop', default='corner', type=str,
        help='Spatial crop method in training (random | corner | center)')
    parser.add_argument('--no_hflip', action='store_true',
        help='Disable horizontal flipping. Consider enabling for hand gestures '
             'where left/right handedness matters.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int,
        help='1: inputs in [0-255]. 255: inputs in [0-1].')
    parser.add_argument('--adap_temp', action='store_true',
        help='Adaptive temporal cropping when input frames exceed sample_duration')
    parser.set_defaults(adap_temp=False)

    # -------------------------------------------------------------------------
    # Pretrained / resume weights
    # During finetune, point these at the Jester checkpoint.
    # Leave empty for pretrain or scratch modes.
    # -------------------------------------------------------------------------
    parser.add_argument('--pretrain_path', default='', type=str,
        help='Path to pretrained model (.pth). Used in finetune mode.')
    parser.add_argument('--pretrain_path_det', default='', type=str,
        help='Path to pretrained detector model (.pth). Used in finetune mode.')
    parser.add_argument('--pretrain_path_clf', default='', type=str,
        help='Path to pretrained classifier model (.pth). Used in finetune mode.')
    parser.add_argument('--pretrain_dataset', default='jester', type=str,
        choices=['jester', ''],
        help='Dataset the pretrained model was trained on')
    parser.add_argument('--resume_path', default='', type=str,
        help='Path to a mid-run checkpoint to resume from (.pth)')
    parser.add_argument('--resume_path_det', default='', type=str)
    parser.add_argument('--resume_path_clf', default='', type=str)

    # -------------------------------------------------------------------------
    # Fine-tuning control
    # ft_begin_index: which residual block to start unfreezing from.
    # 0 = unfreeze everything, 4 = only last block (recommended starting point
    # for Jester -> IPN). Lower if validation plateaus.
    # -------------------------------------------------------------------------
    parser.add_argument('--ft_begin_index', default=4, type=int,
        help='Block index to start fine-tuning from. 0=all layers, 4=last block only.')

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # Recommended: lr=0.1 for pretrain/scratch, lr=0.001 for finetune
    # -------------------------------------------------------------------------
    parser.add_argument('--learning_rate', default=0.1, type=float,
        help='Initial learning rate. Use ~0.1 for pretrain/scratch, ~0.001 for finetune.')
    parser.add_argument('--lr_steps', default=[10, 20, 30, 40, 100], type=float,
        nargs='+', metavar='LRSteps',
        help='Epochs at which to decay learning rate by 10')
    parser.add_argument('--lr_patience', default=10, type=int,
        help='Patience of ReduceLROnPlateau scheduler')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--dampening', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--nesterov', action='store_true')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--begin_epoch', default=1, type=int)
    parser.add_argument('--n_val_samples', default=3, type=int)
    parser.add_argument('--checkpoint', default=10, type=int)

    # -------------------------------------------------------------------------
    # Model architecture — detector
    # -------------------------------------------------------------------------
    parser.add_argument('--model_det', default='resnet', type=str,
        help='Detector model (resnet | preresnet | wideresnet | resnext | densenet)')
    parser.add_argument('--model_depth_det', default=18, type=int,
        help='Detector resnet depth (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut_det', default='B', type=str)
    parser.add_argument('--wide_resnet_k_det', default=2, type=int)
    parser.add_argument('--resnext_cardinality_det', default=32, type=int)
    parser.add_argument('--no_first_lay_det', action='store_true',
        help='If true, first conv layer is replaced')
    parser.set_defaults(no_first_lay_det=False)

    # -------------------------------------------------------------------------
    # Model architecture — classifier
    # -------------------------------------------------------------------------
    parser.add_argument('--model_clf', default='resnet', type=str,
        help='Classifier model (resnet | preresnet | wideresnet | resnext | densenet)')
    parser.add_argument('--model_depth_clf', default=18, type=int,
        help='Classifier resnet depth (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut_clf', default='B', type=str)
    parser.add_argument('--wide_resnet_k_clf', default=2, type=int)
    parser.add_argument('--resnext_cardinality_clf', default=32, type=int)
    parser.add_argument('--no_first_lay_clf', action='store_true',
        help='If true, first conv layer is replaced')
    parser.set_defaults(no_first_lay_clf=False)

    # Shared model args (used when det and clf share the same backbone)
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--model_depth', default=18, type=int)
    parser.add_argument('--resnet_shortcut', default='B', type=str)
    parser.add_argument('--wide_resnet_k', default=2, type=int)
    parser.add_argument('--resnext_cardinality', default=32, type=int)
    parser.add_argument('--width_mult', default=1.0, type=float,
        help='Width multiplier to scale number of filters')

    # -------------------------------------------------------------------------
    # Validation / test settings
    # -------------------------------------------------------------------------
    parser.add_argument('--no_train', action='store_true')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=True)
    parser.add_argument('--scale_in_test', default=1.0, type=float)
    parser.add_argument('--crop_position_in_test', default='c', type=str,
        help='Crop position in test (c | tl | tr | bl | br)')
    parser.add_argument('--no_softmax_in_test', action='store_true')
    parser.set_defaults(no_softmax_in_test=False)

    # -------------------------------------------------------------------------
    # Detector / classifier inference strategy
    # -------------------------------------------------------------------------
    parser.add_argument('--det_strategy', default='raw', type=str,
        help='Detector smoothing filter (raw | median | ma | ewma)')
    parser.add_argument('--det_queue_size', default=1, type=int)
    parser.add_argument('--det_counter', default=1, type=float,
        help='Consecutive detections required before triggering')
    parser.add_argument('--clf_strategy', default='raw', type=str,
        help='Classifier smoothing filter (raw | median | ma | ewma)')
    parser.add_argument('--clf_queue_size', default=1, type=int)
    parser.add_argument('--clf_threshold_pre', default=1, type=float,
        help='Cumulative sum threshold for early prediction')
    parser.add_argument('--clf_threshold_final', default=1, type=float,
        help='Cumulative sum threshold for final prediction')
    parser.add_argument('--clf_threshold', default=0.1, type=float,
        help='Score threshold below which output is treated as no-gesture')
    parser.add_argument('--stride_len', default=1, type=int,
        help='Stride length of the sliding window video loader')

    # -------------------------------------------------------------------------
    # Misc
    # -------------------------------------------------------------------------
    parser.add_argument('--no_cuda', action='store_true')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=8, type=int)
    parser.add_argument('--manual_seed', default=1, type=int)

    args = parser.parse_args()
    return args
