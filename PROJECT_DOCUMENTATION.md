# IPN Hand: Real-Time Continuous Hand Gesture Recognition

## What is the Project?

**IPN Hand** is a video dataset and benchmark framework for **Real-Time Continuous Hand Gesture Recognition (HGR)**. This project contains the PyTorch implementation, training/testing code, and pre-trained models for evaluating 3D Convolutional Neural Network (3D-CNN) architectures on the task of hand gesture recognition.

The goal of this project is to recognize dynamic hand gestures continuously from RGB video streams. It tackles both:
1.  **Isolated Gesture Recognition** (Offline): Classifying a single pre-segmented clip of a gesture.
2.  **Continuous Gesture Detection** (Online): Detecting "when" a gesture occurs and classifying "what" the gesture is in real-time, long continuous video streams. 

The dataset and evaluation models focus on 13 specific hand gestures intended for controlling pointers and interacting with touchless screens (e.g., pointing, clicking, zooming, throwing), plus one "non-gesture" background class. 

## Repository Structure

The codebase is structured to separate data loading, model architecture, training loops, and test executions:

- **Core Execution Scripts**:
  - `main.py`: The central entry point for **offline** (batch) experiments, including training from scratch, fine-tuning, and evaluating on isolated gesture clips.
  - `online_test.py`: Implements the **online** detection pipeline, designed to process continuous video streams and detect/classify gestures in real-time using a two-stage approach (Detector + Classifier).
  - `offline_test.py`: Used for batch evaluation of models on isolated clips.
  - `train.py` & `validation.py` & `test.py`: Contain the core logic for the training iterations, validation metrics, and model inference loops.
  - `opts.py`: Defines all command-line arguments and configurations (e.g., hyper-parameters, data paths, smoothing strategies, model selection).

- **Data Pipeline (`datasets/` & `utils/`)**:
  - `datasets/`: Includes specific PyTorch `Dataset` loaders for the IPN dataset (`ipn.py`, `ipn_online.py`), Jester dataset (`jester.py`), and NVGesture dataset (`nv.py`).
  - `utils/`: Contains preprocessing scripts (`ipn_prepare.py`) and tools for generating JSON annotations (`ipn_json.py`) required by the dataset loaders.
  - `spatial_transforms.py` & `temporal_transforms.py`: Manage video-specific augmentations like multi-scale cropping, temporal jittering, adaptive cropping, and channel normalization.

- **Model Architectures (`models/`)**:
  Contains implementations of 3D-CNN backbones utilized for spatiotemporal feature extraction:
  - Standard Heavy 3D-CNNs: `resnet.py`, `resnetl.py`, `resnext.py`, and `c3d.py`.
  - Lightweight Edge Models: `mobilenetv2.py` and `shufflenetv2.py`.
  - `model.py` (Root): A factory script that initializes these models based on the configurations provided in `opts.py`.

- **Annotations and Pretrained Weights**:
  - `annotation_ipnGesture/`: Contains the ground truth labels in JSON and TXT formats for the various dataset splits.
  - `report_ipn/`: Intended directory to store pre-trained PyTorch model checkpoints (`.pth` files) for benchmarking or resuming training.
  - `tests/`: Contains bash scripts (`.sh`) demonstrating how to run the CLI tools for training, offline testing, and online testing.

## Prerequisites & Requirements

Ensure you have the following installed:
- Python 3.5+
- PyTorch 1.0+
- TorchVision
- Pillow
- OpenCV

## How to Use

### 1. Preparation
1. **Download the Dataset**: Download the IPN Hand dataset from the [official project page](https://gibranbenitez.github.io/IPN_Hand/).
2. **Setup Pretrained Models**: Download the pre-trained weights (e.g., ResNeXt-101 or ResNet-50 models) and store them in the `./report_ipn/` directory.

### 2. Offline Training and Testing (Isolated Gestures)
The offline mode evaluates or trains the model on pre-clipped videos where one gesture is performed per clip. 
- You can refer to the shell scripts in the `tests/` directory for exact configurations.
- For **Isolated Testing**, update the dataset paths in `./tests/run_offline_ipn_Clf.sh` and execute:
  ```bash
  bash tests/run_offline_ipn_Clf.sh
  ```
- Behind the scenes, `main.py` is invoked with arguments like `--mode finetune`, `--dataset ipn`, `--model resnetl`, and various transformation arguments (found in `opts.py`).

### 3. Online Testing (Continuous Gestures)
The online pipeline simulates a real-time environment by feeding a continuous stream of frames into a dual-model system (a Detector to see if a gesture is happening, and a Classifier to determine which one).
- To run the **Continuous Testing**, update the dataset paths in `./tests/run_online_ipnTest.sh` and execute:
  ```bash
  bash tests/run_online_ipnTest.sh
  ```
- This triggers `online_test.py`, which supports various smoothing strategies (e.g., Exponential Weighted Moving Average - EWMA) to prevent flickering predictions in a live video feed. It also supports an optional MediaPipe backend (`--det_backend mediapipe`) for faster frame-level hand presence detection.

## Dataset Details
The IPN Hand dataset consists of 5,649 RGB video instances containing continuous gestures. There are 13 interactive gesture classes and 1 non-gesture class (D0X). It operates natively at 640x480 resolution and 30 frames per second (fps). 
