import cv2
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model, _modify_first_conv_layer
from mean import get_mean, get_std
from spatial_transforms import Compose, Scale, CenterCrop, ToTensor, Normalize
from utils import Queue

CLASS_LABELS = ["None", "GESTURE ACTIVE!"]


def load_cnn_detector(opt):
    opt.resume_path = opt.resume_path_det
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.modality = opt.modality_det
    opt.resnet_shortcut = opt.resnet_shortcut_det
    opt.n_classes = 2
    opt.n_finetune_classes = 2
    opt.no_first_lay = opt.no_first_lay_det

    opt.arch = f'{opt.model}-{opt.model_depth}'
    detector, _ = generate_model(opt)

    if opt.resume_path:
        print(f'Loading detector: {opt.resume_path}')
        checkpoint = torch.load(opt.resume_path, map_location='cpu', weights_only=False)

        if 'module.conv1.weight' in checkpoint['state_dict']:
            ckpt_w = checkpoint['state_dict']['module.conv1.weight']
            m = detector.module if hasattr(detector, 'module') else detector
            if ckpt_w.shape[1] != m.conv1.weight.shape[1] or ckpt_w.shape[2] != m.conv1.weight.shape[2]:
                detector = _modify_first_conv_layer(detector, ckpt_w.shape[2], ckpt_w.shape[1])

        detector.load_state_dict(checkpoint['state_dict'], strict=False)

    detector.eval()
    if not opt.no_cuda:
        detector = detector.cuda()
    return detector


def run_camera():
    opt = parse_opts_online()
    opt.no_cuda = not torch.cuda.is_available()
    opt.batch_size = 1

    # Defaults for the detector checkpoint shipped with the repo
    opt.resume_path_det = 'report_ipn/ipnDet_sc8b64_resnetl-10.pth'
    opt.sample_duration_det = 8
    opt.model_det = 'resnetl'
    opt.model_depth_det = 10
    opt.resnet_shortcut_det = 'A'
    opt.norm_value = 1

    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    spatial_transform = Compose([
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value),
        Normalize(opt.mean, opt.std),
    ])

    if opt.det_backend == 'mediapipe':
        from mediapipe_detector import MediaPipeDetector
        mp_det = MediaPipeDetector(min_detection_confidence=opt.mediapipe_confidence)
        cnn_det = None
        backend_label = f'MediaPipe (conf≥{opt.mediapipe_confidence})'
    else:
        cnn_det = load_cnn_detector(opt)
        mp_det = None
        backend_label = f'CNN ({opt.model_det}-{opt.model_depth_det})'

    myqueue = Queue(8, n_classes=2)
    clip_buffer = []  # only used by CNN path

    cap = cv2.VideoCapture(0)
    print(f'\n[SYSTEM] Gesture Detector Started — backend: {backend_label}')
    print('[SYSTEM] Mirroring camera feed. Press q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for intuitive use
        confidence = 0.0
        current_status = 'None'
        color = (0, 0, 255)

        if opt.det_backend == 'mediapipe':
            outputs = mp_det.detect(frame)
            myqueue.enqueue(outputs.tolist())
            smoothed = myqueue.ma
            pred_idx = int(np.argmax(smoothed))
            confidence = float(smoothed[pred_idx])
            if pred_idx == 1 and confidence > 0.70:
                current_status = 'GESTURE ACTIVE!'
                color = (0, 255, 0)

        else:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            clip_buffer.append(spatial_transform(img))
            if len(clip_buffer) > opt.sample_duration_det:
                clip_buffer.pop(0)

            if len(clip_buffer) == opt.sample_duration_det:
                clip = torch.stack(clip_buffer, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
                if not opt.no_cuda:
                    clip = clip.cuda()

                with torch.no_grad():
                    m = cnn_det.module if hasattr(cnn_det, 'module') else cnn_det
                    if clip.shape[1] < m.conv1.in_channels:
                        pad = torch.zeros(
                            1, m.conv1.in_channels - 3, opt.sample_duration_det, 112, 112,
                            device=clip.device,
                        )
                        clip = torch.cat([clip, pad], dim=1)

                    outputs = F.softmax(cnn_det(clip), dim=1).cpu().numpy()[0]
                    myqueue.enqueue(outputs.tolist())
                    smoothed = myqueue.ma
                    pred_idx = int(np.argmax(smoothed))
                    confidence = float(smoothed[pred_idx])
                    if pred_idx == 1 and confidence > 0.70:
                        current_status = 'GESTURE ACTIVE!'
                        color = (0, 255, 0)

        print(f'\rConf: {confidence:.2f} | {current_status}    ', end='')

        cv2.putText(frame, f'Status: {current_status}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Conf: {confidence:.2f}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f'Det: {backend_label}', (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1)

        cv2.imshow('IPN Hand Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if mp_det:
        mp_det.close()


if __name__ == '__main__':
    run_camera()
