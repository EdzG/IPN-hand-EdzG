#!/bin/bash
# Train the binary gesture DETECTOR on IPN-Hand from scratch.
# Output checkpoint: results_ipn/ipnDet_sc8b64_resnetl-10_best.pth
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0

python "$REPO_ROOT/main.py" \
	--ipn_root_path "$REPO_ROOT" \
	--ipn_video_path src/datasets/HandGestures/IPN_dataset \
	--ipn_annotation_path annotation_ipnGesture/ipnbinary.json \
	--result_path "$REPO_ROOT/results_ipn" \
	--dataset ipn \
	--mode scratch \
	--model resnetl \
	--model_depth 10 \
	--resnet_shortcut A \
	--n_classes 2 \
	--n_finetune_classes 2 \
	--sample_duration 16 \
	--batch_size 64 \
	--learning_rate 0.01 \
	--lr_steps 30 60 90 120 \
	--n_epochs 150 \
	--n_threads 4 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
	--weighted \
	--store_name ipnDet_sc8b64
