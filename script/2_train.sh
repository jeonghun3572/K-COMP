## Example of running the training script

accelerate launch \
--main_process_port  \
--num_processes  \
--num_machines  \
--mixed_precision  \
--dynamo_backend  \
../code/train.py \
--model_id "google/gemma-2b" \
--per_device_train_batch_size  \
--per_device_eval_batch_size  \
--batch_size  \
--lr_scheduler_type  \
--learning_rate  \
--num_train_epochs  \
--warmup_ratio  \
--train_data  \
--eval_data  \
--output_dir  \
--log_name  \
--wandb_run_name  \