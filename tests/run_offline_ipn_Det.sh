#!/bin/bash
# Offline evaluation of the DETECTOR checkpoint on the IPN-Hand test split.
# Requires: results_ipn/ipnDet_sc8b64_resnetl-10_best.pth (produced by run_det_ipn_trainRel-sc8b64.sh)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0

python "$REPO_ROOT/offline_test.py" \
	--ipn_root_path "$REPO_ROOT" \
	--ipn_video_path src/datasets/HandGestures/IPN_dataset \
	--ipn_annotation_path annotation_ipnGesture/ipnbinary.json \
	--result_path "$REPO_ROOT/results_ipn" \
	--resume_path "$REPO_ROOT/results_ipn/ipnDet_sc8b64_resnetl-10_best.pth" \
	--store_name ipnDet_sc8b64 \
	--dataset ipn \
	--modality RGB \
	--model resnetl \
	--model_depth 10 \
	--resnet_shortcut A \
	--n_classes 2 \
	--n_finetune_classes 2 \
	--sample_duration 8 \
	--batch_size 1 \
	--n_val_samples 1 \
	--test_subset test \
	--no_train \
	--no_val \
	--test
