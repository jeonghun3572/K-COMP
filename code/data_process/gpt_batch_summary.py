import json
import argparse


def template(passage, question):
    return f"""
Please extract the content about the entity in fewer than four sentences.

### Passage
{passage.strip()}

### Entity
{question}
""".strip()


def main(args):
    with open(args.input_path, encoding="utf-8") as f:
        json_data = json.load(f)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for idx, data in enumerate(json_data):
            request = {
                "custom_id": f"{args.split}-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [{"role": "user", "content": template(data['passage'], data['question'])}],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output batch request JSONL file.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for summarization.")
    parser.add_argument("--split", type=str, default="train", help="Prefix of custom_id.")
    args = parser.parse_args()
    main(args)
