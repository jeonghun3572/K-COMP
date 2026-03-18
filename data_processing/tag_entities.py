import pandas as pd
import json
import parmap
import argparse


def my_task(chunks):
    json_data, df = chunks
    total = []

    for data in json_data:
        entity_description = []
        entities = []

        if "entity" in data:
            for entity in data['entity']:
                index_list = df.index[df['title'] == entity.lower()].tolist()
                if index_list:
                    description = df['description'][index_list[0]]
                    text = f"{entity}: {description}<eod>"
                    entity_description.append(text.strip())
                    entities.append(entity)

        temp = {
            'question': data['question'],
            'answer': data['answer'],
            'passage': data['passage'],
            'entity': data.get('entity', []),
        }

        if entity_description:
            temp.update({
                'entity_include_prompt': entities,
                'description': "\n".join(entity_description).strip()
            })

        if "summary" in data:
            temp['summary'] = data['summary']

        total.append(temp)

    return total


def main(args):
    with open(args.desc_data_wiki, encoding="utf-8") as f:
        desc_data_wiki = json.load(f)
    with open(args.desc_data_med, encoding="utf-8") as f:
        desc_data_med = json.load(f)

    desc_data = {
        'title': [data['title'].strip().lower() for data in desc_data_wiki + desc_data_med],
        'description': [data['description'].strip() for data in desc_data_wiki + desc_data_med]
    }
    df = pd.DataFrame(desc_data)

    with open(args.input_path, encoding="utf-8") as f:
        json_data = json.load(f)

    num_cores = 55
    json_chunks = [json_data[i::num_cores] for i in range(num_cores)]
    chunks = [(chunk, df) for chunk in json_chunks]

    results = parmap.map(my_task, chunks, pm_pbar=True, pm_processes=num_cores)
    total = [item for sublist in results for item in sublist]

    with open(args.output_path, 'w', encoding="UTF-8") as f:
        json.dump(total, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('--desc_data_wiki', type=str, required=True, help='Path to the wiki description data')
    parser.add_argument('--desc_data_med', type=str, required=True, help='Path to the medical description data')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output JSON file')
    args = parser.parse_args()
    main(args)
