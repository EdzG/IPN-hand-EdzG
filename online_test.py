import time
import os
import sys
import json
import csv
import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from pathlib import Path

from src.opts import parse_opts_online
from src.model import generate_model, _modify_first_conv_layer
from src.mean import get_mean, get_std
from src.transforms import *
from src.transforms.target_transforms import ClassLabel
from src.dataset import get_online_data
from src.utils import Logger, AverageMeter, LevenshteinDistance, Queue


def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))


def load_models(opt):
    """
    Load classifier (always) and CNN detector (only when det_backend == 'cnn').
    Returns (detector_or_None, classifier).
    """
    result_dir = Path(opt.result_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)
    torch.manual_seed(opt.manual_seed)

    detector = None

    # --- Detector (CNN path only) ---
    if opt.det_backend == 'cnn':
        opt.resume_path  = opt.resume_path_det
        opt.pretrain_path = opt.pretrain_path_det
        opt.sample_duration = opt.sample_duration_det
        opt.model         = opt.model_det
        opt.model_depth   = opt.model_depth_det
        opt.modality      = opt.modality_det
        opt.resnet_shortcut = opt.resnet_shortcut_det
        opt.n_classes     = opt.n_classes_det
        opt.n_finetune_classes = opt.n_finetune_classes_det
        opt.no_first_lay  = opt.no_first_lay_det
        opt.arch = f'{opt.model}-{opt.model_depth}'

        with open(result_dir / f'opts_det_{opt.store_name}.json', 'w') as f:
            json.dump(vars(opt), f)

        detector, _ = generate_model(opt)

        if opt.resume_path:
            print(f'Loading detector checkpoint: {opt.resume_path}')
            checkpoint = torch.load(opt.resume_path, weights_only=False)
            assert opt.arch == checkpoint['arch']
            detector.load_state_dict(checkpoint['state_dict'])

        print('Detector:\n', detector)
        print('Detector params:', sum(p.numel() for p in detector.parameters() if p.requires_grad))

    # --- Classifier ---
    opt.resume_path   = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model         = opt.model_clf
    opt.model_depth   = opt.model_depth_clf
    opt.modality      = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes     = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    opt.no_first_lay  = opt.no_first_lay_clf
    opt.arch = f'{opt.model}-{opt.model_depth}'

    with open(result_dir / f'opts_clf_{opt.store_name}.json', 'w') as f:
        json.dump(vars(opt), f)

    classifier, _ = generate_model(opt)

    if opt.resume_path:
        print(f'Loading classifier checkpoint: {opt.resume_path}')
        checkpoint = torch.load(opt.resume_path, weights_only=False)
        assert opt.arch == checkpoint['arch']
        classifier.load_state_dict(checkpoint['state_dict'])

    print('Classifier:\n', classifier)
    print('Classifier params:', sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    return detector, classifier


def main():
    opt = parse_opts_online()
    opt.store_name = f'{opt.store_name}_{opt.dataset}_{opt.model_clf}'

    # Per-dataset path resolution (mirrors offline_test.py)
    if opt.dataset == 'ipn':
        root = Path(opt.ipn_root_path)
        opt.video_path      = str(root / opt.ipn_video_path)
        opt.annotation_path = str(root / opt.ipn_annotation_path)

    detector, classifier = load_models(opt)
    sys.stdout.flush()

    # MediaPipe detector (when selected)
    mp_det = None
    if opt.det_backend == 'mediapipe':
        from src.mediapipe_detector import MediaPipeDetector
        mp_det = MediaPipeDetector(min_detection_confidence=opt.mediapipe_confidence)
        print(f'[INFO] Using MediaPipe detector (conf≥{opt.mediapipe_confidence})')

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value),
        norm_method,
    ])
    target_transform = ClassLabel()

    # Build list of test video paths
    assert opt.dataset == 'ipn', 'online_test.py only supports the IPN dataset'
    file_set = os.path.join(opt.video_path, 'Video_TestList.txt')
    test_paths = []
    with open(file_set, 'rb') as f:
        for line in f:
            vid_name = line.decode().strip().split('\t')[0]
            test_paths.append(os.path.join(opt.video_path, 'frames', vid_name))

    print('Start Evaluation')
    if detector is not None:
        detector.eval()
    classifier.eval()

    levenshtein_accuracies = AverageMeter()
    det_idxs       = []
    end_frames     = []
    pre_classes    = []
    all_pred_frames  = []
    all_pred_starts  = []
    all_pred         = []
    all_true_frames  = []
    all_true_starts  = []
    all_true         = []

    for videoidx, path in enumerate(test_paths, start=1):
        opt.whole_path = os.path.join('frames', path.split(os.sep)[-1])

        active_index     = 0
        passive_count    = 0
        active           = False
        prev_active      = False
        finished_prediction = None
        pre_predict      = False

        cum_sum          = np.zeros(opt.n_classes_clf)
        clf_selected_queue = np.zeros(opt.n_classes_clf)
        det_selected_queue = np.zeros(opt.n_classes_det)
        myqueue_det = Queue(opt.det_queue_size, n_classes=opt.n_classes_det)
        myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)

        print(f'[{videoidx}/{len(test_paths)}] {path}')
        sys.stdout.flush()

        opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)
        test_data = get_online_data(opt, spatial_transform, None, target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
        )

        results    = []
        pred_frames  = []
        pred_start   = []
        pred_starts  = []
        prev_best1   = opt.n_classes_clf
        det_idx      = np.zeros(6000)
        end_fra      = np.zeros(1000)
        pre_cla      = np.zeros(1000)
        det_idx[0]   = test_data.data[-1]['frame_indices'][-1]

        for i, (inputs, targets) in enumerate(test_loader):
            if not opt.no_cuda:
                targets = targets.cuda(non_blocking=True)

            with torch.no_grad():

                # --- Detection ---
                if opt.det_backend == 'mediapipe':
                    frame_idx  = test_data.data[i]['frame_indices'][-1]
                    frame_path = os.path.join(path, f'{frame_idx:05d}.jpg')
                    frame_bgr  = cv2.imread(frame_path)
                    outputs_det = (
                        mp_det.detect(frame_bgr) if frame_bgr is not None
                        else np.array([1.0, 0.0], dtype=np.float32)
                    )
                else:
                    inputs_det  = inputs[:, :, -opt.sample_duration_det:, :, :]
                    outputs_det = detector(inputs_det)
                    outputs_det = F.softmax(outputs_det, dim=1)
                    outputs_det = outputs_det.cpu().numpy()[0].reshape(-1)

                myqueue_det.enqueue(outputs_det.tolist())

                if opt.det_strategy == 'raw':
                    det_selected_queue = outputs_det
                elif opt.det_strategy == 'median':
                    det_selected_queue = myqueue_det.median
                elif opt.det_strategy == 'ma':
                    det_selected_queue = myqueue_det.ma
                elif opt.det_strategy == 'ewma':
                    det_selected_queue = myqueue_det.ewma

                prediction_det = np.argmax(det_selected_queue)

                if prediction_det == 1:
                    det_idx[test_data.data[i]['frame_indices'][-1]] = 1
                    pred_start.append(test_data.data[i]['frame_indices'][-1])

                    inputs_clf  = inputs
                    outputs_clf = classifier(inputs_clf)
                    outputs_clf = F.softmax(outputs_clf, dim=1)
                    outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1)

                    myqueue_clf.enqueue(outputs_clf.tolist())
                    passive_count = 0

                    if opt.clf_strategy == 'raw':
                        clf_selected_queue = outputs_clf
                    elif opt.clf_strategy == 'median':
                        clf_selected_queue = myqueue_clf.median
                    elif opt.clf_strategy == 'ma':
                        clf_selected_queue = myqueue_clf.ma
                    elif opt.clf_strategy == 'ewma':
                        clf_selected_queue = myqueue_clf.ewma
                else:
                    outputs_clf = np.zeros(opt.n_classes_clf)
                    myqueue_clf.enqueue(outputs_clf.tolist())
                    passive_count += 1

            if passive_count >= opt.det_counter:
                active = False
            else:
                active = True

            if active:
                active_index += 1
                cum_sum = (
                    (cum_sum * (active_index - 1))
                    + (weighting_func(active_index) * clf_selected_queue)
                ) / active_index

                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre:
                    finished_prediction = True
                    pre_predict = True
            else:
                active_index = 0

            if not active and prev_active:
                finished_prediction = True
            elif active and not prev_active:
                finished_prediction = False

            if test_data.data[i]['frame_indices'][-1] % 500 == 0:
                print(f'No gestures at frame {test_data.data[i]["frame_indices"][-1]}')
                sys.stdout.flush()

            if finished_prediction:
                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if cum_sum[best1] > opt.clf_threshold_final:
                    if pre_predict:
                        if best1 != prev_best1 and cum_sum[best1] > opt.clf_threshold_final:
                            results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                            print(f'Early  class={best1} prob={cum_sum[best1]:.3f} '
                                  f'frames {pred_start[0]}~{test_data.data[i]["frame_indices"][-1]}')
                            pred_frames.append(test_data.data[i]['frame_indices'][-1])
                            pred_starts.append(pred_start[0])
                            pred_start = []
                    else:
                        if cum_sum[best1] > opt.clf_threshold_final:
                            if best1 == prev_best1:
                                if cum_sum[best1] > 5:
                                    results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                                    print(f'Late   class={best1} prob={cum_sum[best1]:.3f} '
                                          f'frames {pred_start[0]}~{test_data.data[i]["frame_indices"][-1]}')
                                    pred_frames.append(test_data.data[i]['frame_indices'][-1])
                                    pred_starts.append(pred_start[0])
                                    pred_start = []
                            else:
                                results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                                print(f'Late   class={best1} prob={cum_sum[best1]:.3f} '
                                      f'frames {pred_start[0]}~{test_data.data[i]["frame_indices"][-1]}')
                                pred_frames.append(test_data.data[i]['frame_indices'][-1])
                                pred_starts.append(pred_start[0])
                                pred_start = []

                    finished_prediction = False
                    prev_best1 = best1
                    pred_start = []

                cum_sum      = np.zeros(opt.n_classes_clf)
                active_index = 0
                pred_start   = []
                sys.stdout.flush()

            if not active and prev_active:
                pre_predict = False
            prev_active = active

        # --- Ground truth for this video ---
        true_classes = []
        true_frames  = []
        true_starts  = []
        val_list_path = os.path.join(os.path.dirname(opt.annotation_path), 'vallistall.txt')
        with open(val_list_path) as csvfile:
            for row in csv.reader(csvfile, delimiter=' '):
                if row[0][2:] == opt.whole_path and row[1] != '1':
                    true_classes.append(int(row[1]) - 2)
                    true_starts.append(int(row[2]))
                    true_frames.append(int(row[3]))

        true_classes = np.array(true_classes)
        if not results:
            predicted  = np.array(results)
            pred_frames_arr = np.array(pred_frames)
            levenshtein_distance = -1
        else:
            pred_frames_arr = np.array(pred_frames)
            predicted = np.array(results)[:, 1]
            levenshtein_distance = LevenshteinDistance(true_classes, predicted)
            levenshtein_accuracy = 1 - (levenshtein_distance / len(true_classes))
            pre_cla[: len(predicted)] = predicted + 1
            end_fra[: len(pred_frames_arr)] = pred_frames_arr

        if levenshtein_distance < 0:
            levenshtein_accuracies.update(0, len(true_classes))
        else:
            levenshtein_accuracies.update(levenshtein_accuracy, len(true_classes))

        pred    = []
        all_pred.append(predicted.tolist())
        all_pred_frames.append(pred_frames_arr.tolist() if hasattr(pred_frames_arr, 'tolist') else pred_frames)
        all_pred_starts.append(pred_starts)
        for k, pn in enumerate(predicted):
            pred.append(f'{pn}({pred_starts[k]}~{pred_frames[k]})')

        true_gt = []
        all_true.append(true_classes.tolist())
        all_true_frames.append(true_frames)
        all_true_starts.append(true_starts)
        for k, pn in enumerate(true_classes):
            true_gt.append(f'{pn}({true_starts[k]}~{true_frames[k]})')

        print('Predicted:', ' '.join(pred) if pred else 'NONE')
        print('True:     ', ' '.join(true_gt))
        print(f'Levenshtein acc = {levenshtein_accuracies.val:.3f} (avg {levenshtein_accuracies.avg:.3f}) '
              f'det frames {int(np.sum(det_idx[2:]))}/{int(det_idx[0])}')
        det_idxs.append(det_idx)
        end_frames.append(end_fra)
        pre_classes.append(pre_cla)
        sys.stdout.flush()

    print(f'Average Levenshtein Accuracy = {levenshtein_accuracies.avg:.4f}')
    print('----- Evaluation finished -----')

    res_data = {
        'all_pred':        all_pred,
        'all_pred_frames': all_pred_frames,
        'all_pred_starts': all_pred_starts,
        'all_true':        all_true,
        'all_true_frames': all_true_frames,
        'all_true_starts': all_true_starts,
    }
    result_dir = Path(opt.result_path)
    with open(result_dir / f'res_{opt.store_name}.json', 'w') as dst_file:
        json.dump(res_data, dst_file)

    if mp_det:
        mp_det.close()


if __name__ == '__main__':
    main()
