import os
import json
import logging

import torch
import pyarrow.csv as pc

import src.normalize_text

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        normalize=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
    ):
        self.normalize_fn = src.normalize_text.normalize if normalize else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break
        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break
        return examples, counter

    def __getitem__(self, index):
        example = self.data[index]
        question_key = "question_1" if "question_1" in example else "question"
        prompt_input = (
            f"### Question\n{example[question_key]}\n\n"
            f"### Passage\n{example['passage']}\n\n"
            f"### Entity\n"
        )
        return {
            "question": self.normalize_fn(example["question"]),
            "answer": self.normalize_fn(example["answer"]),
            "prompt_input": self.normalize_fn(prompt_input.strip()),
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        question = [ex["question"] for ex in batch]
        answer = [ex["answer"] for ex in batch]
        prompt_input = [ex["prompt_input"] for ex in batch]

        p_out = self.tokenizer.batch_encode_plus(
            prompt_input,
            max_length=7936,
            truncation=True,
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
        )

        return {
            "question": question,
            "answer": answer,
            "p_out": p_out,
        }


def load_passages(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Passages file not found: {path}")
    logger.info(f"Loading passages from: {path}")

    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for line in fin:
                passages.append(json.loads(line))

    if not passages:
        reader = pc.read_csv(path, parse_options=pc.ParseOptions(delimiter="\t"))
        df = reader.to_pandas()
        for row in df.itertuples(index=False, name=None):
            if row[0] != "id":
                passages.append({"id": int(row[0]), "title": row[1], "text": row[2]})

    return passages
