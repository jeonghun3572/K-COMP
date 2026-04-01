#!/bin/bash
# Step 3: Inference
# Run from the project root directory.

uv run python inference.py \
    --model_path <MODEL_PATH> \
    --per_device_test_batch_size <BATCH_SIZE> \
    --dataset <DATASET_NAME> \
    --test_data <TEST_DATA_PATH> \
    --log_name <LOG_NAME>
