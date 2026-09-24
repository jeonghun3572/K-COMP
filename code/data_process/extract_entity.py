import re
import json
import nltk
import spacy
import argparse

from tqdm import tqdm
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS


def process_question(question, wlem, nlp):
    tb = TextBlob(question)
    question_temp = [wlem.lemmatize(word_token).strip() for word_token in tb.words]
    question = ' '.join(question_temp)
    return nlp(question)


def extract_entities(doc_q, stop_words_list):
    seen = set()
    entities = []
    for ent_q in doc_q.ents:
        ent_q_text = ent_q.text.lower()
        if ent_q_text not in seen and ent_q_text not in stop_words_list:
            entities.append(ent_q.text)
            seen.add(ent_q_text)
            for k in ent_q_text.split(' '):
                seen.add(k)
    return entities


def filter_entities(entities, passage):
    return [ent for ent in entities if re.search(re.escape(ent), passage, re.I)]


def main(args):
    stop_words_list = [word.lower() for word in STOP_WORDS]
    wlem = nltk.WordNetLemmatizer()
    spacy.require_gpu(args.gpu)
    nlp = spacy.load(args.spacy_model)

    with open(args.input_path, "r", encoding="UTF-8") as f:
        json_data = [json.loads(line) for line in f]

    total = []
    for data in tqdm(json_data, desc="Processing data"):
        passage = "\n\n".join(f"{ctx['title']}\n{ctx['text']}" for ctx in data['ctxs'][:args.n_docs])

        doc_q = process_question(data['question'], wlem, nlp)
        entities = extract_entities(doc_q, stop_words_list)
        f_entity = filter_entities(entities, passage)

        if f_entity:
            total.append({
                "question": data["question"],
                "answer": data["answer"],
                "passage": passage,
                "entity": f_entity,
                "ctxs": [{"title": ctx["title"].strip(), "text": ctx["text"].strip()} for ctx in data["ctxs"]],
            })

    with open(args.output_path, 'w', encoding="UTF-8") as f:
        json.dump(total, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--n-docs", type=int, default=5, help="Number of retrieved passages to use")
    parser.add_argument("--spacy-model", type=str, default="en_core_sci_scibert", help="spaCy model for NER")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for spaCy")
    args = parser.parse_args()
    main(args)
