import json
import re
import os
import argparse

def process_data(input_path, output_path):
    with open(input_path) as f:
        json_data = json.load(f)

    total = []
    patterns = [
        re.compile(r"(<ent>)s\b"),
        re.compile(r"(<ent>)es\b"),
        re.compile(r"(<ent>)ies\b"),
        re.compile(r"(<ent>)ves\b")
    ]

    for data in json_data:
        question = data['question']
        temp = {
            'question': data['question'],
            'answer': data['answer'],
            'passage': data['passage'],
            'summary': data.get('summary', ''),
            'entity': data.get('entity', []),
            'entity_include_prompt': data.get('entity_include_prompt', []),
            'description': data.get('description', '')
        }

        entity_list = data.get('entity_include_prompt', data.get('entity', []))
        if entity_list:
            pattern = re.compile('|'.join([re.escape(word) for word in entity_list]), re.IGNORECASE)
            question_include_entity = pattern.sub('<ent>', question)
            for pat in patterns:
                question_include_entity = pat.sub(r"\1", question_include_entity)
            temp['question_include_entity'] = question_include_entity

            ## Make dataset for question masking
            try:
                temp["prompt"] = f"### Question\n{data['question_include_entity']}\n\n### Passage\n{data['passage']}"
                temp["completion"] = f"### Entity\n{data['description']}\n\n### Summary\n{data['summary']}"
            except:
                temp["prompt"] = f"### Question\n{data['question']}\n\n### Passage\n{data['passage']}"
                temp["completion"] = f"### Entity\nNone\n\n### Summary\n{data['summary']}"
            total.append(temp)

    with open(output_path, 'w', encoding="UTF-8") as f:
        json.dump(total, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Process JSON data.")
    parser.add_argument('input_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('output_path', type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    if os.path.exists(args.input_path):
        process_data(args.input_path, args.output_path)
    else:
        print(f"Input file {args.input_path} does not exist.")

if __name__ == "__main__":
    main()
