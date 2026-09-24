import os
import pickle
import argparse
import torch
import torch.nn.functional as F
import src.data
import src.normalize_text

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_passages(args, passages, model, tokenizer):
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []

    with torch.no_grad():
        for k, p in enumerate(tqdm(passages)):
            batch_ids.append(p["id"])
            text = p["title"] + " " + p["text"] if not args.no_title and "title" in p else p["text"]
            text = text.lower() if args.lowercase else text
            text = src.normalize_text.normalize(text) if args.normalize_text else text
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.to(model.device) for k, v in encoded_batch.items()}
                outputs = model(**encoded_batch)
                embeddings = mean_pooling(outputs, encoded_batch['attention_mask'])
                embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
                embeddings = embeddings[:, :args.projection_size]
                embeddings = F.normalize(embeddings, p=2, dim=1)

                allids.extend(batch_ids)
                allembeddings.append(embeddings.cpu())
                batch_ids, batch_text = [], []

    return allids, torch.cat(allembeddings, dim=0).numpy()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.cuda()
    model.eval()

    passages = src.data.load_passages(args.passages)
    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = len(passages) if args.shard_id == args.num_shards - 1 else start_idx + shard_size
    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f"{args.prefix}_{args.shard_id:02d}")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)
    print(f"Saved {len(allids)} passage embeddings to {save_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Path to the retriever model.")
    parser.add_argument("--passages", type=str, required=True, help="Path to passages (.tsv or .jsonl file)")
    parser.add_argument("--output-dir", type=str, default="embeddings", help="Directory path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="Prefix for saved embeddings")
    parser.add_argument("--shard-id", type=int, default=0, help="ID of the current shard")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--per-gpu-batch-size", type=int, default=512, help="Batch size for the passage encoder forward pass")
    parser.add_argument("--passage-maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument("--projection-size", type=int, default=512, help="Embedding dimension to keep")
    parser.add_argument("--no-title", action="store_true", help="Do not add title to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before encoding")
    parser.add_argument("--normalize-text", action="store_true", help="Normalize text before encoding")

    args = parser.parse_args()
    main(args)
