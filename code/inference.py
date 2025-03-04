import os
import re
import json
import torch
import argparse

from accelerate import Accelerator
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import src
from src.data import Dataset, Collator

def main(args):
    torch.manual_seed(args.seed)
    output_dir = f"result/{args.dataset}/summ"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    special_tokens_dict = {'additional_special_tokens': ['<ent>', '<eod>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side='left'

    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )

    collator = Collator(tokenizer)
    test_dataset = Dataset(datapaths=args.test_data, normalize=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.per_device_test_batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    accelerator = Accelerator()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.eval()

    total = []
    pattern = r'(.*?)### Summary.*?(?=###)'

    for batch in test_dataloader:
        question = batch['question']
        answer = batch['answer']
        p_out = batch['p_out']

        with torch.no_grad():
            pred_s_ids = model.generate(**p_out, generation_config=gen_config)
            pred_summary = tokenizer.batch_decode(pred_s_ids[:, len(p_out['input_ids'][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        for j in range(len(question)):
            temp = {}
            temp['question'] = question[j]
            temp['answer'] = answer[j]
            text = pred_summary[j].strip()
            text = text.replace("\n<|im_start|>assistant", "")
            text = text.replace("<|im_start|>user", "")
            text = text.replace("<|im_start|>", "")
            text = text.replace("<|im_end|>", "")
            if "### Entity" in text:
                text = text.partition("### Entity")[0]
                text = text.strip()
            text = f"### Entity\n{text}\n\n"

            try:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    extracted_text = match.group(0)
                else:
                    pattern_initial = r'.*?### Summary.*?\n\n'
                    match_initial = re.match(pattern_initial, text, re.DOTALL)
                    extracted_text = match_initial.group(0) if match_initial else text
                extracted_text = extracted_text.strip()

                try:
                    paragraphs = re.split(r'\n\n', extracted_text)
                    lines = paragraphs[0].split("\n")
                    entity = "\n".join(lines[1:]).strip()
                    lines = paragraphs[1].split("\n")
                    summary = "\n".join(lines[1:]).strip()

                    if "<|" in summary:
                        summary = summary.partition("<|")[0]
                        summary = summary.strip()
                    if "<|" in entity:
                        entity = entity.partition("<|")[0]
                        entity = entity.strip()
                except:
                    split_text = re.split(r'### Summary', extracted_text)
                    entity = split_text[0].replace("### Entity\n", "")
                    summary = split_text[1].replace("### Summary\n", "")

                    if "<|" in summary:
                        summary = summary.partition("<|")[0]
                        summary = summary.strip()
                    if "<|" in entity:
                        entity = entity.partition("<|")[0]
                        entity = entity.strip()

                temp['summary'] = summary.strip()
                temp['description'] = entity.strip()

            except:
                entity = text.strip()
                summary = text.strip()
                if "<|" in summary:
                    summary = summary.partition("<|")[0]
                    summary = summary.strip()
                if "<|" in entity:
                    entity = entity.partition("<|")[0]
                    entity = entity.strip()
                temp['summary'] = summary.strip()
                temp['description'] = ""
            total.append(temp)

    with open(f"{output_dir}/{args.log_name}.jsonl", encoding="utf-8", mode="w") as f:
        for i in total:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="t5-small")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--per_device_test_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_name", type=str, required=True)

    args = parser.parse_args()

    main()
