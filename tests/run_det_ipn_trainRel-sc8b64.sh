#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main.py \
	--root_path . \
	--video_path datasets/HandGestures/IPN_dataset \
	--annotation_path annotation_ipnGesture/ipnbinary.json \
	--result_path results_ipn \
	--dataset ipn \
	--sample_duration 8 \
	--learning_rate 0.01 \
	--model resnetl \
	--model_depth 10 \
	--resnet_shortcut A \
	--batch_size 64 \
	--n_classes 2 \
	--n_finetune_classes 2 \
	--n_threads 32 \
	--checkpoint 1 \
	--modality RGB \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
	--n_epochs 150 \
	--lr_steps 30 60 90 120 \
	--store_name ipnDet_sc8b64