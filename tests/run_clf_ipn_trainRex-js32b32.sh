#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
	--root_path . \
	--video_path datasets/HandGestures/IPN_dataset \
	--annotation_path annotation_ipnGesture/ipnall_but_None.json \
	--result_path results_ipn \
	--dataset ipn \
	--sample_duration 32 \
	--learning_rate 0.01 \
	--model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 32 \
	--n_classes 13 \
	--n_finetune_classes 13 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
	--n_epochs 100 \
	--store_name ipnClf_jes32r_b32