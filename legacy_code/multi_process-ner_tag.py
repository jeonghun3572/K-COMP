import sys
import spacy
import parmap
import json
import pdb
import re
from transformers import AutoTokenizer
from itertools import permutations

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def generate_all_permutations(input_string):
    words = input_string.split()
    permutations_list = list(permutations(words))
    return [' '.join(permutation) for permutation in permutations_list]

def subfinder(mylist, pattern, first_only=False):
    matches_indx = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches_indx.append((i, i+len(pattern)))
            if first_only:
                break
    return matches_indx

def update_labels(labels, matched_ranges):
    for range_i in matched_ranges:
        if labels[range_i[0]] == 0:
            labels[range_i[0]] = 1  # [B]eginning
            for i in range(range_i[0]+1, range_i[1]):
                labels[i] = 2  # [I]nside
    return labels

def update_bio_labels_below(labels, source, match_patterns, tokens, tokenizer, first_only=False):
    entity = []
    for pattern in match_patterns:
        flag = False
        if " " in pattern:
            all_pattern = generate_all_permutations(pattern)
            for pat in all_pattern:
                found = re.findall(re.escape(pat), source, re.IGNORECASE)
                for matched in found:
                    matched_space = ' ' + matched if matched[0] != ' ' else matched
                    matched_ids = tokenizer.encode(matched, add_special_tokens=False)
                    matched_ranges = subfinder(tokens, matched_ids, first_only)
                    matched_ids_space = tokenizer.encode(matched_space, add_special_tokens=False)
                    matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
                    labels = update_labels(labels, matched_ranges)
                    labels = update_labels(labels, matched_ranges_space)
                    flag = True
        else:
            found = re.findall(re.escape(pattern), source, re.IGNORECASE)
            for matched in found:
                matched_space = ' ' + matched if matched[0] != ' ' else matched
                matched_ids = tokenizer.encode(matched, add_special_tokens=False)
                matched_ranges = subfinder(tokens, matched_ids, first_only)
                matched_ids_space = tokenizer.encode(matched_space, add_special_tokens=False)
                matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
                labels = update_labels(labels, matched_ranges)
                labels = update_labels(labels, matched_ranges_space)
                flag = True

        if flag:
            entity.append(pattern)

    match_patterns = list(set(re.findall(r'<obj>(.*?)</obj>', source)))
    for pattern in match_patterns:
        flag = False
        found = re.findall(rf'\b{re.escape(pattern)}\b', source, re.IGNORECASE)
        for matched in found:
            matched_space = ' ' + matched if matched[0] != ' ' else matched
            matched_ids = tokenizer.encode(matched, add_special_tokens=False)
            matched_ranges = subfinder(tokens, matched_ids, first_only)
            matched_ids_space = tokenizer.encode(matched_space, add_special_tokens=False)
            matched_ranges_space = subfinder(tokens, matched_ids_space, first_only)
            labels = update_labels(labels, matched_ranges)
            labels = update_labels(labels, matched_ranges_space)
            flag = True
        
        if flag:
            entity.append(pattern)

    f_entity = []
    seen = set()
    for ent in entity:
        if ent.lower() not in seen:
            f_entity.append(ent)
            seen.add(ent.lower())

    return labels, f_entity

def process_data(json_data, tokenizer, nlp, mask=True):
    total = []
    for data in json_data:
        question = data['question']
        answer = data['answer']
        passage = data['passage'] if mask else data['passage2']
        summary_tag = data['summary']
        entity = data['entity']

        doc_q = [nlp(question), nlp(answer)]
        for docq in doc_q:
            for ent_q in docq.ents:
                entity.append(ent_q.text)

        t_entity = []
        seen = set()
        for ent in entity:
            if ent.lower() not in seen:
                t_entity.append(ent)
                seen.add(ent.lower())

        tokens = tokenizer.encode(passage, max_length=4096, truncation=True)
        labels = [0] * len(tokens)

        labels, f_entity = update_bio_labels_below(labels, passage, t_entity, tokens, tokenizer, first_only=False)
        assert len(tokens) == len(labels)

        temp = {
            'question': question,
            'answer': answer,
            'passage_mask': data['passage'],
            'passage_nomask': data['passage2'],
            'summary_tag': summary_tag,
            'summary_notag': summary_tag.replace("<obj>", "").replace("</obj>", "").replace("<ref>", "").replace("</ref>", "").strip(),
            'entity': entity,
            'entity_tag': f_entity,
            'ner_labels': labels
        }
        total.append(temp)

    return total

def my_task_nomask(chunks):
    nlp = spacy.load("en_core_sci_lg")
    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-x-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<obj>', '</obj>', '<ref>', '</ref>']})
    return process_data(chunks[0], tokenizer, nlp, mask=False)

def my_task_mask(chunks):
    nlp = spacy.load("en_core_sci_lg")
    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-x-large')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<obj>', '</obj>', '<ref>', '</ref>']})
    return process_data(chunks[0], tokenizer, nlp, mask=True)

def main():
    num_cores = 80

    with open('datasets/medquad-5_tag_real/medquad-train-sum.json') as f:
        json_data = json.load(f)
    json_chunk = [json_data[i::num_cores] for i in range(num_cores)]
    chunks = [[chunk] for chunk in json_chunk]
    results = parmap.map(my_task_mask, chunks, pm_pbar=True, pm_processes=num_cores)
    total = [item for sublist in results for item in sublist]

    with open("datasets/medquad-bart_train/medquad-train-masking-ansner.jsonl", encoding="utf-8", mode="w") as f:
        for item in total:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
