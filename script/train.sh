## Example of running the training script

accelerate launch \
    --main_process_port  \
    --num_processes  \
    --num_machines  \
    --mixed_precision  \
    --dynamo_backend  \
    ../code/train.py \
    --model-id "google/gemma-2b" \
    --train-data  \
    --eval-data  \
    --output-dir  \
    --per-device-train-batch-size  \
    --per-device-eval-batch-size  \
    --batch-size  \
    --lr-scheduler-type  \
    --learning-rate  \
    --num-train-epochs  \
    --warmup-ratio  \
    --wandb-run-name
