import os
import re
import json
import torch
import argparse

from accelerate import Accelerator
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.data import Dataset, Collator

CHAT_TOKENS = ["\n<|im_start|>assistant", "<|im_start|>user", "<|im_start|>", "<|im_end|>"]


def parse_output(text):
    for token in CHAT_TOKENS:
        text = text.replace(token, "")
    if "### Entity" in text:
        text = text.partition("### Entity")[0].strip()
    text = f"### Entity\n{text}\n\n"

    match = re.search(r'.*?### Summary.*?(?=###)', text, re.DOTALL) or re.match(r'.*?### Summary.*?\n\n', text, re.DOTALL)
    extracted = (match.group(0) if match else text).strip()

    paragraphs = extracted.split("\n\n")
    if len(paragraphs) > 1:
        description = "\n".join(paragraphs[0].split("\n")[1:])
        summary = "\n".join(paragraphs[1].split("\n")[1:])
    elif "### Summary" in extracted:
        description, _, summary = extracted.partition("### Summary")
        description = description.replace("### Entity\n", "")
    else:
        description, summary = "", extracted

    return description.partition("<|")[0].strip(), summary.partition("<|")[0].strip()


def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    special_tokens_dict = {'additional_special_tokens': ['<ent>', '<eod>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "left"

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )

    test_dataset = Dataset(args.test_data, normalize=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.per_device_test_batch_size,
        num_workers=args.num_workers,
        collate_fn=Collator(tokenizer, max_length=args.max_length),
        drop_last=False,
    )

    accelerator = Accelerator()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.eval()

    total = []
    for batch in test_dataloader:
        p_out = batch['p_out']
        with torch.no_grad():
            pred_ids = model.generate(**p_out, generation_config=gen_config)
        preds = tokenizer.batch_decode(pred_ids[:, p_out['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        for question, answer, pred in zip(batch['question'], batch['answer'], preds):
            description, summary = parse_output(pred.strip())
            total.append({
                "question": question,
                "answer": answer,
                "summary": summary,
                "description": description,
            })

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in total:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--per-device-test-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=7936, help="Maximum prompt length.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
