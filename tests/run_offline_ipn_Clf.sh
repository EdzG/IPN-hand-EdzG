#!/bin/bash
# Offline evaluation of the CLASSIFIER checkpoint on the IPN-Hand test split.
# Requires: results_ipn/ipnClf_jes32r_b32_resnext-101_best.pth (produced by run_clf_ipn_trainRex-js32b32.sh)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0

python "$REPO_ROOT/offline_test.py" \
	--ipn_root_path "$REPO_ROOT" \
	--ipn_video_path src/datasets/HandGestures/IPN_dataset \
	--ipn_annotation_path annotation_ipnGesture/ipnall_but_None.json \
	--result_path "$REPO_ROOT/results_ipn" \
	--resume_path "$REPO_ROOT/results_ipn/ipnClf_jes32r_b32_resnext-101_best.pth" \
	--store_name ipnClf_jes32r_b32 \
	--dataset ipn \
	--modality RGB \
	--model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--n_classes 13 \
	--n_finetune_classes 13 \
	--sample_duration 32 \
	--batch_size 1 \
	--n_val_samples 1 \
	--test_subset test \
	--no_train \
	--no_val \
	--test
