import json
import torch
import pyarrow.csv as pc

from src import normalize_text


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, normalize=False):
        self.normalize_fn = normalize_text.normalize if normalize else lambda x: x
        with open(data_path, encoding="utf-8") as f:
            if data_path.endswith(".jsonl"):
                self.data = [json.loads(line) for line in f]
            else:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example['question_1'] if "question_1" in example else example['question']
        prompt_input = f"### Question\n{question}\n\n### Passage\n{example['passage']}\n\n### Entity"
        return {
            "question": self.normalize_fn(example['question']),
            "answer": self.normalize_fn(example['answer']),
            "prompt_input": self.normalize_fn(prompt_input),
        }


class Collator(object):
    def __init__(self, tokenizer, max_length=7936):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        p_out = self.tokenizer.batch_encode_plus(
            [ex['prompt_input'] for ex in batch],
            max_length=self.max_length,
            truncation=True,
            padding='longest',
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {
            "question": [ex['question'] for ex in batch],
            "answer": [ex['answer'] for ex in batch],
            "p_out": p_out,
        }


def load_passages(path):
    print(f"Loading passages from: {path}")
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    df = pc.read_csv(path, parse_options=pc.ParseOptions(delimiter="\t")).to_pandas()
    return [{"id": int(row[0]), "title": row[1], "text": row[2]} for row in df.itertuples(index=False, name=None)]
