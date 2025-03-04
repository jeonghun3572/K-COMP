import re
import sys
import json
import nltk
import spacy
import parmap
import multiprocessing
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def preprocess_text(data, desc_data, remove_set, remove_desc1, remove_desc2, nlp):
    tb = TextBlob(data['question'])
    wlem = nltk.WordNetLemmatizer()
    question_temp = [wlem.lemmatize(word_token).strip() for word_token in tb.words]
    question = ' '.join(question_temp)
    entity2 = []
    entity_prompt = ""
    seen = set()

    for desc in desc_data:
        title = desc['title']
        description = desc['description']
        flag = True

        if (title.lower() in set(remove_set)) or (remove_desc1 in description) or (remove_desc2 in description) or (title.lower() in seen):
            flag = False

        if flag and re.findall(rf'\b{re.escape(title)}\b', question, re.I):
            doc_title = nlp(title)
            if len(doc_title.ents) > 0:
                entity_prompt = f"{entity_prompt}\n{title}: {description}"
                entity2.append(title)
                seen.add(title.lower())
                for k in title.split(' '):
                    seen.add(k.lower())

    temp = {
        'question': data['question'],
        'answer': data['answer'],
        'passage': data['passage'],
        'summary': data['summary'],
        'entity': data['entity'],
        'entity_tag': entity2,
        'summary_entity': f"{data['summary'].strip()}\n\n### Entity\n{entity_prompt.strip()}".strip(),
        'passage_notitle': data.get('passage_notitle', ''),
        'passage_top1': data.get('passage_top1', ''),
        'passage_top10': data.get('passage_top10', ''),
        'passage_top15': data.get('passage_top15', ''),
        'passage_top20': data.get('passage_top20', '')
    }
    return temp

def my_task(chunks):
    stop_words_list = [word.lower() for word in STOP_WORDS]
    remove_set = [
        'people', 'disease', 'responsive', 'dominant', 'older', 'risk', 'treatments', 'giant', 'multi', 'use', 'glass', 'adult', 'type', 'gas', 'triple',
        'need', 'research', 'family', 'history', 'causes', 'failure', 'blue', 'person', 'hour', 'sleep', 'form', 'human'
    ] + stop_words_list
    remove_desc1 = "refer to:"
    remove_desc2 = "alphabet"
    json_data, desc_data = chunks
    nlp = spacy.load("en_core_sci_lg")
    total = [preprocess_text(data, desc_data, remove_set, remove_desc1, remove_desc2, nlp) for data in json_data]
    return total

def my_task2(chunks):
    stop_words_list = [word.lower() for word in STOP_WORDS]
    remove_set = [
        'people', 'disease', 'responsive', 'dominant', 'older', 'risk', 'treatments', 'giant', 'multi', 'use', 'glass', 'adult', 'type', 'gas', 'triple',
        'need', 'research', 'family', 'history', 'causes', 'failure', 'blue', 'person', 'hour', 'sleep', 'form', 'human', 'individual', 'context'
    ] + stop_words_list
    remove_desc1 = "refer to:"
    remove_desc2 = "alphabet"
    json_data, desc_data = chunks
    nlp = spacy.load("en_core_sci_lg")
    total = [preprocess_text(data, desc_data, remove_set, remove_desc1, remove_desc2, nlp) for data in json_data]
    return total

def load_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def process_dataset(file_path, output_path, desc_data, task_function, num_cores):
    json_data = load_data(file_path)
    json_chunk = [json_data[i::num_cores] for i in range(num_cores)]
    chunks = [[chunk, desc_data] for chunk in json_chunk]
    results = parmap.map(task_function, chunks, pm_pbar=True, pm_processes=num_cores)
    total = [item for sublist in results for item in sublist]
    save_data(output_path, total)

def main():
    desc_data_wiki = load_data("kilm/data/wiki/short_descriptions_filter.json")
    desc_data_nih = load_data("TopicIndex2.json")

    desc_data = [{'title': data['title'], 'description': data['description']} for data in desc_data_wiki + desc_data_nih]
    desc_data = sorted(desc_data, key=lambda x: len(x['title']), reverse=True)
    num_cores = 60

    datasets = [
        ("datasets/bioasq-3_summary/bioasq-test.json", "datasets/bioasq-4_tag/bioasq-test.json", my_task),
        ("datasets/bioasq-3_summary/bioasq-val.json", "datasets/bioasq-4_tag/bioasq-val.json", my_task),
        ("datasets/bioasq-3_summary/bioasq-train.json", "datasets/bioasq-4_tag/bioasq-train.json", my_task),
        ("datasets/medquad-3_summary/medquad-test-sum.json", "datasets/medquad-5_tag/medquad-test.json", my_task2),
        ("datasets/medquad-3_summary/medquad-val-sum.json", "datasets/medquad-5_tag/medquad-val.json", my_task2),
        ("datasets/medquad-3_summary/medquad-train-sum.json", "datasets/medquad-5_tag/medquad-train.json", my_task2),
        ("datasets/mashqa-3_summary/medquad-test-sum.json", "datasets/mashqa-4_tag/mashqa-test.json", my_task2),
        ("datasets/mashqa-3_summary/medquad-val-sum.json", "datasets/mashqa-4_tag/mashqa-val.json", my_task2),
        ("datasets/mashqa-3_summary/medquad-train-sum.json", "datasets/mashqa-4_tag/mashqa-train.json", my_task2)
    ]

    for input_path, output_path, task_function in datasets:
        process_dataset(input_path, output_path, desc_data, task_function, num_cores)

if __name__ == "__main__":
    main()
