#!/bin/bash
# Smoke test to verify the end-to-end pipeline.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/results_smoke_test"

echo "Starting Smoke Test..."

python "$REPO_ROOT/main.py" \
	--ipn_root_path "$REPO_ROOT" \
	--ipn_video_path src/datasets/HandGestures/IPN_dataset \
	--ipn_annotation_path annotation_ipnGesture/ipn_smoke_test.json \
	--result_path "$REPO_ROOT/results_smoke_test" \
	--dataset ipn \
	--mode scratch \
	--model resnetl \
	--model_depth 10 \
	--n_classes 2 \
	--batch_size 2 \
	--n_epochs 1 \
	--n_threads 1 \
	--checkpoint 1 \
	--no_val \
	--store_name smoke_test_run

echo "Smoke Test Completed Successfully!"
