# IPN Hand Repository Overview

This repository provides a comprehensive PyTorch implementation for the **IPN Hand** dataset, a benchmark for **real-time continuous hand gesture recognition (HGR)**. It is based on the research presented in the paper *"IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition"* (ICPR 2020).

## Repository Structure Overview

### 1. Core Execution Scripts (Root Directory)
*   **`main.py`**: The central entry point for **offline** experiments. It handles training and testing on pre-segmented gesture clips.
*   **`online_test.py`**: Implements the **online** detection pipeline, which processes continuous video streams to detect and classify gestures in real-time.
*   **`offline_test.py`**: Used for batch evaluation of models on isolated gesture clips.
*   **`train.py`, `validation.py`, `test.py`**: Contain the core logic for the training loops, validation metrics, and model inference.
*   **`opts.py`**: Defines all command-line arguments, including separate configurations for offline training and online real-time testing (e.g., smoothing strategies like EWMA).

### 2. Model Architectures (`models/`)
This folder contains the 3D-CNN backbones used for spatiotemporal feature extraction:
*   **Standard 3D-CNNs**: `resnet.py`, `resnext.py`, and `c3d.py`.
*   **Lightweight Models**: `mobilenetv2.py` and `shufflenetv2.py`, optimized for real-time performance on edge devices.
*   **`model.py` (Root)**: A factory script that initializes these models based on the configurations provided in `opts.py`.

### 3. Data Pipeline (`datasets/`, `utils/`, and transforms)
*   **`datasets/`**: Specific loaders for the IPN and NVGesture datasets (`ipn.py`, `ipn_online.py`).
*   **`utils/`**: Pre-processing scripts (`ipn_prepare.py`) and tools to generate the JSON annotation format (`ipn_json.py`) required by the loaders.
*   **`spatial_transforms.py` & `temporal_transforms.py`**: Handle video-specific augmentations such as multi-scale cropping, temporal jittering, and normalization.

### 4. Annotations and Models
*   **`annotation_ipnGesture/`**: Contains the ground truth labels in JSON and TXT formats for various splits (e.g., `ipnall.json`, `trainlistall.txt`).
*   **`report_ipn/`**: Stores pre-trained PyTorch model checkpoints (`.pth` files) used for benchmarking.

---

## Primary Purpose
The repository is designed to facilitate both **isolated gesture recognition** (classifying a single clip) and **continuous gesture detection** (finding "when" and "what" in a long video).

1.  **Benchmarking**: It provides a standard environment to evaluate 3D-CNN architectures on the IPN Hand dataset.
2.  **Multi-Modal Analysis**: Supports training on RGB, Optical Flow, and Semantic Segmentation to improve accuracy while maintaining real-time speeds.
3.  **Real-Time Deployment**: Includes an "online" mode that uses a two-stage approach (Detector + Classifier) to handle continuous video interaction, similar to how a touchless screen interface would operate.
