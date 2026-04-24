#!/bin/bash
# Online (sliding-window) evaluation of the full detector+classifier pipeline.
# Requires both checkpoints produced by the two training scripts:
#   results_ipn/ipnDet_sc8b64_resnetl-10_best.pth
#   results_ipn/ipnClf_jes32r_b32_resnext-101_best.pth
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python "$REPO_ROOT/online_test.py" \
	--ipn_root_path "$REPO_ROOT" \
	--ipn_video_path src/datasets/HandGestures/IPN_dataset \
	--ipn_annotation_path annotation_ipnGesture/ipnall.json \
	--result_path "$REPO_ROOT/results_ipn" \
	--dataset ipn \
	--det_backend cnn \
	--resume_path_det "$REPO_ROOT/results_ipn/ipnDet_sc8b64_resnetl-10_best.pth" \
	--resume_path_clf "$REPO_ROOT/results_ipn/ipnClf_jes32r_b32_resnext-101_best.pth" \
	--model_det resnetl \
	--model_depth_det 10 \
	--resnet_shortcut_det A \
	--n_classes_det 2 \
	--n_finetune_classes_det 2 \
	--sample_duration_det 8 \
	--modality_det RGB \
	--model_clf resnext \
	--model_depth_clf 101 \
	--resnet_shortcut_clf B \
	--n_classes_clf 13 \
	--n_finetune_classes_clf 13 \
	--sample_duration_clf 32 \
	--modality_clf RGB \
	--store_name RGB \
	--batch_size 1 \
	--n_threads 4 \
	--n_val_samples 1 \
	--train_crop random \
	--test_subset test \
	--det_backend cnn \
	--det_strategy ma \
	--det_queue_size 4 \
	--det_counter 2 \
	--clf_strategy ma \
	--clf_queue_size 16 \
	--clf_threshold_pre 0.15 \
	--clf_threshold_final 0.15 \
	--stride_len 1
