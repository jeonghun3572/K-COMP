import os
import argparse
import torch
import wandb
from datasets import load_dataset
from accelerate import PartialState
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback

def formatting_prompts_func(example):
    return [f"{prompt.strip()}\n\n{completion.strip()}" for prompt, completion in zip(example['prompt'], example['completion'])]

def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")

    num_gpus = torch.cuda.device_count()
    grad_accum_steps = max(args.batch_size // args.per_device_train_batch_size // num_gpus, 1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={'': device_string},
        trust_remote_code=True,
    )

    special_tokens_dict = {'additional_special_tokens': ['<ent>', '<eod>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "right"
    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        dataloader_drop_last=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        run_name=args.wandb_run_name,
        report_to="wandb",
        save_only_model=True,
        ddp_find_unused_parameters=False,
        max_seq_length=8192,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--train_data', type=str, required=True, help="Path to the training data.")
    parser.add_argument('--eval_data', type=str, required=True, help="Path to the evaluation data.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory.")
    parser.add_argument('--model_id', type=str, required=True, help="Model ID.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device for training.")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size per device for evaluation.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate.")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler type.")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay.")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="Warmup ratio.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--batch_size', type=int, default=8, help="Total batch size.")
    parser.add_argument('--wandb_run_name', type=str, default="run", help="WandB run name.")
    
    args = parser.parse_args()
    main(args)
