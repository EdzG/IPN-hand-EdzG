#!/bin/bash
# Train the 13-class gesture CLASSIFIER on IPN-Hand from scratch.
# Output checkpoint: results_ipn/ipnClf_jes32r_b32_resnext-101_best.pth
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$REPO_ROOT/main.py" \
	--ipn_root_path "$REPO_ROOT" \
	--ipn_video_path src/datasets/HandGestures/IPN_dataset \
	--ipn_annotation_path annotation_ipnGesture/ipnall_but_None.json \
	--result_path "$REPO_ROOT/results_ipn" \
	--dataset ipn \
	--mode scratch \
	--model mobilenetv2 \
	--n_classes 13 \
	--n_finetune_classes 13 \
	--sample_duration 32 \
	--batch_size 8 \
	--learning_rate 0.01 \
	--n_epochs 100 \
	--n_threads 1 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
	--store_name ipnClf_mobv2_b8
