import copy
import os
import sys
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from argparse import Namespace
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import functional as F

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.model import generate_model, _modify_first_conv_layer
from src.mean import get_mean, get_std
from src.setup import resolve_dataset_paths
from src.dataset import get_test_set
from src.transforms.spatial_transforms import Compose, Scale, CenterCrop, ToTensor, Normalize
from src.transforms.temporal_transforms import TemporalCenterCrop
from src.transforms.target_transforms import ClassLabel
from src.utils import Queue

# Configuration
opt = Namespace(
    # General Options
    no_cuda=not torch.cuda.is_available(),
    norm_value=1,
    batch_size=1,
    manual_seed=1,
    n_threads=0, # Set to 0 for Jupyter compatibility 
    
    # Dataset Paths
    dataset='ipn',
    ipn_root_path=repo_root,
    ipn_video_path='src/datasets/HandGestures/IPN_dataset',
    ipn_annotation_path='annotation_ipnGesture/ipnall_but_None.json',
    jester_root_path='', jester_video_path='', jester_annotation_path='',
    
    # Detector Configuration (CNN Path)
    det_backend='cnn', # Change to 'mediapipe' for lightweight inference
    resume_path_det=os.path.join(repo_root, 'results_ipn/ipnDet_sc8b64_resnetl-10_best.pth'),
    sample_duration_det=8,
    model_det='resnetl',
    model_depth_det=10,
    resnet_shortcut_det='A',
    modality_det='RGB',
    n_classes_det=2,
    n_finetune_classes_det=2,
    no_first_lay_det=False,
    
    # Classifier Configuration
    resume_path_clf=os.path.join(repo_root, 'results_ipn/ipnClf_jes32r_b32_resnext-101_best.pth'),
    sample_duration_clf=32,
    model_clf='resnext',
    model_depth_clf=101,
    resnet_shortcut_clf='B',
    modality_clf='RGB',
    n_classes_clf=13,
    n_finetune_classes_clf=13,
    no_first_lay_clf=False,
    
    # Preprocessing
    mean_dataset='ipn',
    no_mean_norm=False,
    std_norm=False,
    sample_size=112,
    
    # Smoothing / Inference logic
    det_strategy='ma',
    det_queue_size=4,
    det_counter=2,
    clf_strategy='ma',
    clf_queue_size=16,
    clf_threshold_pre=0.15,
    clf_threshold_final=0.15,
    
    # Placeholders for underlying engine functions
    pretrain_path_det='', pretrain_path_clf='', resume_path='',
    scales=[1.0], n_scales=5, scale_step=0.84089641525, initial_scale=1.0,
    store_name='RGB', test_subset='test', true_valid=False
)

# Initialize paths and spatial transformations
resolve_dataset_paths(opt)
opt.mean = get_mean(opt.norm_value)
opt.std = get_std(opt.norm_value)

spatial_transform = Compose([
    Scale(opt.sample_size),
    CenterCrop(opt.sample_size),
    ToTensor(opt.norm_value),
    Normalize(opt.mean, opt.std),
])

# Load Models
def _get_smoothed(queue, strategy):
    return queue.ewma if strategy == 'ewma' else queue.ma

def load_cnn_detector(opt):
    o = copy.copy(opt)
    o.resume_path = opt.resume_path_det
    o.sample_duration = opt.sample_duration_det
    o.model = opt.model_det
    o.model_depth = opt.model_depth_det
    o.modality = opt.modality_det
    o.resnet_shortcut = opt.resnet_shortcut_det
    o.n_classes = opt.n_classes_det
    o.n_finetune_classes = opt.n_finetune_classes_det
    o.no_first_lay = opt.no_first_lay_det
    o.arch = f'{o.model}-{o.model_depth}'

    detector, _ = generate_model(o)

    if os.path.exists(o.resume_path):
        checkpoint = torch.load(o.resume_path, map_location='cpu', weights_only=False)
        if 'module.conv1.weight' in checkpoint['state_dict']:
            ckpt_w = checkpoint['state_dict']['module.conv1.weight']
            m = detector.module if hasattr(detector, 'module') else detector
            if ckpt_w.shape[1] != m.conv1.weight.shape[1] or ckpt_w.shape[2] != m.conv1.weight.shape[2]:
                detector = _modify_first_conv_layer(detector, ckpt_w.shape[2], ckpt_w.shape[1])
        detector.load_state_dict(checkpoint['state_dict'], strict=False)

    detector.eval()
    if not opt.no_cuda: detector = detector.cuda()
    return detector

def load_classifier(opt):
    o = copy.copy(opt)
    o.resume_path = opt.resume_path_clf
    o.sample_duration = opt.sample_duration_clf
    o.model = opt.model_clf
    o.model_depth = opt.model_depth_clf
    o.modality = opt.modality_clf
    o.resnet_shortcut = opt.resnet_shortcut_clf
    o.n_classes = opt.n_classes_clf
    o.n_finetune_classes = opt.n_finetune_classes_clf
    o.no_first_lay = opt.no_first_lay_clf
    o.arch = f'{o.model}-{o.model_depth}'

    classifier, _ = generate_model(o)

    if os.path.exists(o.resume_path):
        checkpoint = torch.load(o.resume_path, map_location='cpu', weights_only=False)
        if 'module.conv1.weight' in checkpoint['state_dict']:
            ckpt_w = checkpoint['state_dict']['module.conv1.weight']
            m = classifier.module if hasattr(classifier, 'module') else classifier
            if ckpt_w.shape[1] != m.conv1.weight.shape[1] or ckpt_w.shape[2] != m.conv1.weight.shape[2]:
                classifier = _modify_first_conv_layer(classifier, ckpt_w.shape[2], ckpt_w.shape[1])
        classifier.load_state_dict(checkpoint['state_dict'], strict=False)

    classifier.eval()
    if not opt.no_cuda: classifier = classifier.cuda()
    return classifier

print("Loading Detector...")
if opt.det_backend == 'mediapipe':
    from src.mediapipe_detector import MediaPipeDetector
    mp_det = MediaPipeDetector(min_detection_confidence=0.5)
    cnn_det = None
else:
    cnn_det = load_cnn_detector(opt)
    mp_det = None

print("Loading Classifier...")
classifier = load_classifier(opt)
print("Initialization Complete.")

def evaluate_and_plot():
    # Configure temporal cropping specifically for the offline testing of the classifier
    opt.sample_duration = opt.sample_duration_clf 
    temporal_transform = Compose([TemporalCenterCrop(opt.sample_duration)])

    # Load IPN-Hand Test Dataset
    test_data = get_test_set(opt, spatial_transform, temporal_transform, ClassLabel())
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_threads
    )

    y_true = []
    y_pred = []

    print(f"Running offline evaluation on {len(test_data)} samples. This may take a moment...")
    start_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if not opt.no_cuda:
                inputs = inputs.cuda(non_blocking=True)
            
            # Forward pass
            outputs = classifier(inputs)
            outputs = F.softmax(outputs, dim=1)
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(targets.numpy())

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches")

    print(f"Evaluation finished in {time.time() - start_time:.2f} seconds.")

    # Calculate Accuracy Metric
    acc = accuracy_score(y_true, y_pred)
    print(f"--- Critical Information ---")
    print(f"Overall Classifier Accuracy (Top-1): {acc * 100:.2f}%")

    # Generate Confusion Matrix Graph
    class_names = list(test_data.class_names.values()) if hasattr(test_data, 'class_names') else [f"Class {i}" for i in range(opt.n_classes_clf)]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    # Normalize the matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

    ax = plt.subplot()
    sns.heatmap(cm_norm, annot=False, ax=ax, xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    ax.set_xlabel("Predicted Gestures", fontsize=12)
    ax.set_ylabel("True Gestures", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f"Normalized Confusion Matrix\n({opt.model_clf}-{opt.model_depth_clf})", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

def run_pipeline():
    # Smoothing queues
    myqueue_det = Queue(opt.det_queue_size, n_classes=opt.n_classes_det)
    myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
    
    # Buffers to hold recent frames
    det_clip_buffer = []
    clf_clip_buffer = []
    
    active = False
    passive_count = 0
    
    cap = cv2.VideoCapture(0) # Change to an mp4 file path to test on pre-recorded video
    print('\n[SYSTEM] Gesture Pipeline Started. Select the popup window and press "q" to quit.')

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # Mirror for intuition
        
        # --- Stage 1: Detector ---
        is_gesture = False
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transformed_img = spatial_transform(img)
        clf_clip_buffer.append(transformed_img)
        if len(clf_clip_buffer) > opt.sample_duration_clf:
            clf_clip_buffer.pop(0)

        if opt.det_backend == 'mediapipe':
            outputs_det = mp_det.detect(frame)
            myqueue_det.enqueue(outputs_det.tolist())
            smoothed_det = _get_smoothed(myqueue_det, opt.det_strategy)
            if int(np.argmax(smoothed_det)) == 1: is_gesture = True
        else:
            det_clip_buffer.append(transformed_img)
            if len(det_clip_buffer) > opt.sample_duration_det:
                det_clip_buffer.pop(0)

            if len(det_clip_buffer) == opt.sample_duration_det:
                clip = torch.stack(det_clip_buffer, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
                if not opt.no_cuda: clip = clip.cuda()

                with torch.no_grad():
                    m = cnn_det.module if hasattr(cnn_det, 'module') else cnn_det
                    if clip.shape[1] < m.conv1.in_channels:
                        pad = torch.zeros(1, m.conv1.in_channels - 3, opt.sample_duration_det, opt.sample_size, opt.sample_size, device=clip.device)
                        clip = torch.cat([clip, pad], dim=1)

                    outputs_det = F.softmax(cnn_det(clip), dim=1).cpu().numpy()[0]
                    myqueue_det.enqueue(outputs_det.tolist())
                    if int(np.argmax(_get_smoothed(myqueue_det, opt.det_strategy))) == 1: is_gesture = True

        # State transition 
        if is_gesture:
            passive_count = 0
            active = True
        else:
            passive_count += 1
            if passive_count >= opt.det_counter:
                active = False
                
        # --- Stage 2: Classifier ---
        status_text = 'Idle'
        color = (0, 0, 255)
        
        if active:
            status_text = 'Gesture Detected - Classifying...'
            color = (0, 255, 255)
            
            if len(clf_clip_buffer) == opt.sample_duration_clf:
                clip_clf = torch.stack(clf_clip_buffer, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
                if not opt.no_cuda: clip_clf = clip_clf.cuda()
                    
                with torch.no_grad():
                    m = classifier.module if hasattr(classifier, 'module') else classifier
                    if clip_clf.shape[1] < m.conv1.in_channels:
                        pad = torch.zeros(1, m.conv1.in_channels - 3, opt.sample_duration_clf, opt.sample_size, opt.sample_size, device=clip_clf.device)
                        clip_clf = torch.cat([clip_clf, pad], dim=1)

                    outputs_clf = F.softmax(classifier(clip_clf), dim=1).cpu().numpy()[0]
                    myqueue_clf.enqueue(outputs_clf.tolist())
                    smoothed_clf = _get_smoothed(myqueue_clf, opt.clf_strategy)

                    pred_class = int(np.argmax(smoothed_clf))
                    conf_clf = float(smoothed_clf[pred_class])

                    if conf_clf > opt.clf_threshold_pre:
                        status_text = f'Class: {pred_class} (Conf: {conf_clf:.2f})'
                    if conf_clf > opt.clf_threshold_final:
                        color = (0, 255, 0)
        else:
            myqueue_clf.enqueue(np.zeros(opt.n_classes_clf).tolist())
            
        cv2.putText(frame, f'Status: {status_text}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('IPN Hand Pipeline', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if mp_det: mp_det.close()

if __name__ == '__main__':
    # evaluate_and_plot() # Uncomment to run the evaluation and plot first
    # run_pipeline() # Uncomment to start the webcam pipeline
    pass