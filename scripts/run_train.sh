#!/bin/bash
# Step 2: Training
# Run from the project root directory.

accelerate launch \
    --main_process_port <PORT> \
    --num_processes <NUM_GPUS> \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    train.py \
    --model_id "google/gemma-2b" \
    --per_device_train_batch_size <TRAIN_BATCH_SIZE> \
    --per_device_eval_batch_size <EVAL_BATCH_SIZE> \
    --batch_size <TOTAL_BATCH_SIZE> \
    --lr_scheduler_type linear \
    --learning_rate <LR> \
    --num_train_epochs <EPOCHS> \
    --warmup_ratio <WARMUP> \
    --train_data <TRAIN_DATA_PATH> \
    --eval_data <EVAL_DATA_PATH> \
    --output_dir <OUTPUT_DIR> \
    --wandb_run_name <RUN_NAME>
