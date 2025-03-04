import json
import argparse


def template(passage, entity):
    return f"""
Please extract the content about the entity in fewer than four sentences.

### Passage
{passage.strip()}

### Entity
{entity}
""".strip()


def process_data(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        json_data = json.load(f)

    total = []
    for idx, data in enumerate(json_data):
        question = data['question']
        passage = data['passage']
        temp = {
            "custom_id": f"train-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": template(passage, question)
                    }
                ]
            }
        }
        total.append(temp)

    with open(output_path, encoding='utf-8', mode='w') as f:
        for item in total:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process input and output paths for summary generation.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    args = parser.parse_args()
    process_data(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
