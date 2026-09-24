import json
import argparse
from tqdm import tqdm


def load_descriptions(paths):
    descriptions = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for data in json.load(f):
                descriptions.setdefault(data['title'].strip().lower(), data['description'].strip())
    return descriptions


def tag_entity(data, descriptions):
    entities = [entity for entity in data.get('entity', []) if entity.lower() in descriptions]
    temp = {
        'question': data['question'],
        'answer': data['answer'],
        'passage': data['passage'],
        'entity': data.get('entity', []),
    }
    if entities:
        temp['entity_include_prompt'] = entities
        temp['description'] = "\n".join(f"{entity}: {descriptions[entity.lower()]}<eod>".strip() for entity in entities)
    if "summary" in data:
        temp['summary'] = data['summary']
    return temp


def main(args):
    # Wiki descriptions take priority over medical ones for duplicated titles
    descriptions = load_descriptions([args.desc_data_wiki, args.desc_data_med])

    with open(args.input_path, encoding="utf-8") as f:
        json_data = json.load(f)

    # Same output order as the original chunked multiprocessing
    chunks = [json_data[i::args.num_chunks] for i in range(args.num_chunks)]
    total = [tag_entity(data, descriptions) for chunk in tqdm(chunks, desc="Tagging entities") for data in chunk]

    with open(args.output_path, 'w', encoding="utf-8") as f:
        json.dump(total, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc-data-wiki', type=str, required=True, help='Path to the wiki description data')
    parser.add_argument('--desc-data-med', type=str, required=True, help='Path to the medical description data')
    parser.add_argument('--input-path', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output JSON file')
    parser.add_argument('--num-chunks', type=int, default=55, help='Number of chunks, which determines the output order')
    args = parser.parse_args()
    main(args)
