import os
import argparse
import json
import pickle
import time
import glob

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import src.index
import src.slurm
import src.data
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the model output using the attention mask.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_queries(args, queries, model, tokenizer):
    """
    Embed the queries using the provided model and tokenizer.
    """
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():
        for k, q in enumerate(queries):
            q = q.lower()
            q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.to(model.device) for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embedding = mean_pooling(output, encoded_batch['attention_mask'])
                embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
                embedding = embedding[:, :args.projection_size]
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding.cpu())
                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")
    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    """
    Index the encoded data using the provided index and embedding files.
    """
    allids = []
    allembeddings = np.array([])
    for file_path in embedding_files:
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    """
    Add embeddings to the index in batches.
    """
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def add_passages(data, passages, top_passages_and_scores):
    """
    Add passages to the original data based on the top passages and scores.
    """
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[int(doc_id)] for doc_id in results_and_scores[0]]
        d["ctxs"] = [{"title": doc["title"], "text": doc["text"]} for doc in docs]


def load_data(data_path):
    """
    Load data from the specified path.
    """
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for example in fin:
                data.append(json.loads(example))
    return data


def main(args):
    """
    Main function to run the passage retrieval.
    """
    print(f"Loading model from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, safe_serialization=True)
    model.eval()
    model = model.cuda()

    passages = src.data.load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}
    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
    index.deserialize_from(args.passages_embeddings)

    data_paths = glob.glob(args.data)
    for path in data_paths:
        data = load_data(path)
        output_path = os.path.join(args.output_dir, os.path.basename(path))

        queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(args, queries, model, tokenizer)

        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time() - start_time_retrieval:.1f} s.")

        add_passages(data, passage_id_map, top_ids_and_scores)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix")
    parser.add_argument("--n_docs", type=int, default=10, help="Number of documents to retrieve per questions")
    parser.add_argument("--validation_workers", type=int, default=32, help="Number of parallel processes to validate results")
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists")
    parser.add_argument("--model_name_or_path", type=str, help="path to directory containing model weights and config file")
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
