import os
import argparse
import pickle
import torch
import torch.nn.functional as F
import src.slurm
import src.data
import src.normalize_text

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_passages(args, passages, model, tokenizer):
    total = 0
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
                embeddings = embeddings[:, :512]
                embeddings = F.normalize(embeddings, p=2, dim=1)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, safe_serialization=True)
    print(f"Model loaded from {args.model_name_or_path}.", flush=True)
    model.cuda()
    model.eval()

    passages_temp = src.data.load_passages(args.passages)
    shard_size = len(passages_temp) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages_temp)

    passages = passages_temp[start_idx:end_idx]
    del passages_temp
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    save_file = os.path.join(args.output_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default="wikipedia_embeddings", help="Directory path to save embeddings")
    parser.add_argument("--prefix", type=str, default="passages", help="Prefix for saved embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="ID of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass")
    parser.add_argument("--passage_maxlength", type=int, default=512, help="Maximum number of tokens in a passage")
    parser.add_argument("--model_name_or_path", type=str, help="Path to directory containing model weights and config file")
    parser.add_argument("--no_fp16", action="store_true", help="Inference in fp32")
    parser.add_argument("--no_title", action="store_true", help="Do not add title to the passage body")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="Normalize text before encoding")

    args = parser.parse_args()

    src.slurm.init_distributed_mode(args)

    main(args)
