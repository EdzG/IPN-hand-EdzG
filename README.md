# Touchless Presentation Control via Hand Gesture Recognition

This repository implements a real-time continuous hand gesture recognition system designed for touchless interaction, specifically optimized for controlling presentations (PowerPoint, PDF readers, etc.). It leverages state-of-the-art 3D Convolutional Neural Networks (3D-CNNs) and MediaPipe to provide a robust, low-latency interface.

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Supported Gestures](#supported-gestures)
4. [Installation & Setup](#installation--setup)
5. [Training Workflows](#training-workflows)
6. [Real-Time Inference & Control](#real-time-inference--control)
7. [Codebase Walkthrough](#codebase-walkthrough)
8. [Acknowledgements](#acknowledgements)

---

## 🌟 Project Overview
The goal of this project is to enable seamless, touchless navigation of digital content. By recognizing complex dynamic gestures in real-time RGB video streams, the system can trigger keyboard events (e.g., Next Slide, Previous Slide, Zoom) without the need for physical controllers.

**Key Features:**
- **Real-Time Detection:** A two-stage pipeline that first detects hand presence and then classifies the specific gesture.
- **Hybrid Backends:** Supports both 3D-CNN and MediaPipe-based detection for different hardware constraints.
- **Continuous Recognition:** Handles long, unsegmented video streams with temporal smoothing to prevent "flickering" predictions.

---

## 🏗 System Architecture

The project employs a dual-model architecture to achieve high accuracy without sacrificing real-time performance.

### 1. Lightweight Hybrid Pipeline (CNN + CNN)
Designed for low-latency execution on commodity hardware.
- **Detector:** A lightweight **ResNetL-10** (8-frame clip) for fast binary classification (Gesture vs. Non-Gesture).
- **Classifier:** Optimized architectures like **ResNet-18, MobileNetV2, or ShuffleNetV2** (32-frame clip) to balance accuracy with real-time CPU/GPU constraints.

### 2. MediaPipe Integration (Landmarks + CNN)
The most efficient path for standard laptops/PCs.
- **Detector:** Uses **MediaPipe Hands** for per-frame hand presence detection (minimal CPU overhead).
- **Classifier:** A lightweight CNN classifier is triggered only when a hand is detected, significantly reducing "Midas-Touch" errors (unintentional inputs).

---

## 🖐 Supported Gestures
The system is trained on the **IPN-Hand** dataset, which includes 13 interactive gesture classes and 1 background (non-gesture) class:

| ID | Gesture | Presentation Action (Typical) |
|---|---|---|
| **D0X** | No Gesture | Idle / Standby |
| **B0A** | Pointing (1 finger) | Mouse Pointer / Laser |
| **B0B** | Pointing (2 fingers) | Secondary Pointer |
| **G01** | Click (1 finger) | Select / Open |
| **G02** | Click (2 fingers) | Right Click / Menu |
| **G03** | Throw Up | Start Presentation / Fullscreen |
| **G04** | Throw Down | Exit Presentation |
| **G05** | Throw Left | **Next Slide** (Page Down) |
| **G06** | Throw Right | **Previous Slide** (Page Up) |
| **G07** | Open Twice | Toggle Overview |
| **G08** | Double Click (1 finger) | Execute / Open |
| **G09** | Double Click (2 fingers) | Context Menu |
| **G10** | Zoom In | **Zoom In** |
| **G11** | Zoom Out | **Zoom Out** |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- PyTorch 1.5+ (with CUDA support recommended)
- OpenCV, Pillow, MediaPipe, Scikit-learn

### Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/IPN-hand-EdzG.git
   cd IPN-hand-EdzG
   ```

2. **Install Dependencies:**
   ```bash
   pip install torch torchvision opencv-python mediapipe scikit-learn pillow
   ```

3. **Prepare Datasets:**
   - Download the **IPN-Hand** dataset and place it in `src/datasets/HandGestures/IPN_dataset`.
   - Use the scripts in `src/utils/data_prep/` to generate required `.json` annotations.

4. **Download Checkpoints:**
   Place pre-trained `.pth` models in the `report_ipn/` directory.

---

## ⚡ Training Workflows

### 1. Training from Scratch (IPN-Hand)
To train the classifier on the IPN-Hand dataset without pre-training:
```bash
python main.py --mode scratch --dataset ipn --model resnext --model_depth 101 --batch_size 32
```

### 2. Pre-training (Jester) & Fine-tuning (IPN)
Recommended for maximum accuracy.
1. **Pre-train on Jester:**
   ```bash
   python main.py --mode pretrain --dataset jester --model resnext --model_depth 101
   ```
2. **Fine-tune on IPN-Hand:**
   ```bash
   python main.py --mode finetune --dataset ipn --pretrain_path report_ipn/jester_checkpoint.pth --ft_begin_index 4
   ```

---

## 🎮 Real-Time Inference & Control

### Live Webcam Test (Detector Only)
Verify your camera and detection backend:
```bash
# Using CNN Detector
python camera_test.py --det_backend cnn

# Using MediaPipe Detector
python camera_test.py --det_backend mediapipe
```

### Continuous Gesture Recognition (Full Pipeline)
Run the full online evaluation on test video sequences:
```bash
bash tests/run_online_ipnTest.sh
```

### Presentation Control Integration
The `online_test.py` script outputs predictions in real-time. To map these to presentation actions, you can extend the prediction loop with `pyautogui`:
```python
# Example mapping (indices correspond to the gesture ID order)
if best1 == 5:   # G05 (Throw Left)
    pyautogui.press('pagedown')
elif best1 == 6: # G06 (Throw Right)
    pyautogui.press('pageup')
elif best1 == 10: # G10 (Zoom In)
    pyautogui.hotkey('ctrl', '+')
```

---

## 📂 Codebase Walkthrough

- **`main.py`**: Entry point for all training and offline evaluation.
- **`online_test.py`**: The core real-time engine handling model synchronization and smoothing.
- **`src/models/`**: Implementation of 3D-CNN backbones (ResNet, ResNeXt, ShuffleNet, etc.).
- **`src/transforms/`**: Complex spatiotemporal augmentations (Multi-scale cropping, temporal padding).
- **`src/mediapipe_detector.py`**: Wrapper for MediaPipe's hand tracking API.
- **`src/opts.py`**: Centralized configuration and CLI argument parsing.

---

## 🤝 Acknowledgements
This project is based on the research from:
- **IPN Hand Dataset:** Gibran Benitez-Garcia et al. (ICPR 2020).
- **MediaPipe:** Google Open Source.
- **PyTorch 3D-CNN:** Adapted from various spatiotemporal recognition benchmarks.
