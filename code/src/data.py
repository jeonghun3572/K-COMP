import os
import torch
import json
import csv
import logging
import pyarrow.csv as pc

import src
from src import normalize


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        normalize=False,
        global_rank=-1,
        world_size=-1,
        maxload=None
    ):
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
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

    ## TODO
    def __getitem__(self, index):
        example = self.data[index]
        if "question_1" in example:
            prompt_input = f'''
### Question
{example['question_1']}

### Passage
{example['passage']}

### Entity
'''
        else:
            prompt_input = f'''
### Question
{example['question']}

### Passage
{example['passage']}

### Entity
'''
        example={
            "question": self.normalize_fn(example['question']),
            "answer": self.normalize_fn(example['answer']),
            "prompt_input": self.normalize_fn(prompt_input.strip())
        }
        return example



class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        question = [ex['question'] for ex in batch]
        answer = [ex['answer'] for ex in batch]
        prompt_input = [ex['prompt_input'] for ex in batch]

        p_out = self.tokenizer.batch_encode_plus(
            prompt_input,
            max_length=7936,
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            return_tensors="pt",
        )

        batch={
            "question": question,
            "answer": answer,
            "p_out": p_out,
        }
        return batch



# Used for passage retrieval
def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)

        # else:
        #     reader = csv.reader(fin, delimiter="\t")
        #     for k, row in enumerate(reader):
        #         if not row[0] == "id":
        #             ex = {"id": row[0], "title": row[1], "text": row[2]}
        #             passages.append(ex)
    if passages == []:
        reader = pc.read_csv(path, parse_options=pc.ParseOptions(delimiter="\t"))
        df = reader.to_pandas()
        for row in df.itertuples(index=False, name=None):
            if row[0] != "id":
                ex = {"id": int(row[0]), "title": row[1], "text": row[2]}
                passages.append(ex)
    return passages
