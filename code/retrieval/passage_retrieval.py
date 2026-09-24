import os
import json
import glob
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import src.data
import src.index
import src.normalize_text

from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_queries(args, queries, model, tokenizer):
    embeddings, batch_question = [], []
    with torch.no_grad():
        for k, q in enumerate(queries):
            batch_question.append(src.normalize_text.normalize(q.lower()))

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


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    index.index_data(ids[:end_idx], embeddings[:end_idx])
    return embeddings[end_idx:], ids[end_idx:]


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for file_path in embedding_files:
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as f:
            ids, embeddings = pickle.load(f)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)


def add_passages(data, passages, top_ids_and_scores):
    assert len(data) == len(top_ids_and_scores)
    for d, (doc_ids, _) in zip(data, top_ids_and_scores):
        d["ctxs"] = [{"title": passages[doc_id]["title"], "text": passages[doc_id]["text"]} for doc_id in doc_ids]


def load_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        return json.load(f)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.cuda()
    model.eval()

    # Load a serialized faiss index directory, or build the index from generate_passage_embeddings.py outputs
    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)
    if os.path.isdir(args.passages_embeddings):
        index.deserialize_from(args.passages_embeddings)
    else:
        embedding_files = sorted(glob.glob(args.passages_embeddings))
        index_encoded_data(index, embedding_files, args.indexing_batch_size)
        if args.save_index:
            index.serialize(os.path.dirname(embedding_files[0]))

    passages = {str(p["id"]): p for p in src.data.load_passages(args.passages)}

    os.makedirs(args.output_dir, exist_ok=True)
    for data_path in glob.glob(args.data):
        data = load_data(data_path)
        questions_embedding = embed_queries(args, [ex["question"] for ex in data], model, tokenizer)

        start_time = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time() - start_time:.1f} s.")

        add_passages(data, passages, top_ids_and_scores)
        output_path = os.path.join(args.output_dir, os.path.basename(data_path))
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Path to the retriever model.")
    parser.add_argument("--data", type=str, required=True, help="Glob path to question data (.json or .jsonl)")
    parser.add_argument("--passages", type=str, required=True, help="Path to passages (.tsv or .jsonl file)")
    parser.add_argument("--passages-embeddings", type=str, required=True, help="Directory of a serialized index, or glob path to encoded passages")
    parser.add_argument("--output-dir", type=str, required=True, help="Results are written to output_dir with data file name")
    parser.add_argument("--n-docs", type=int, default=10, help="Number of documents to retrieve per question")
    parser.add_argument("--per-gpu-batch-size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--question-maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--projection-size", type=int, default=768, help="Embedding dimension to keep")
    parser.add_argument("--save-index", action="store_true", help="Save the index built from encoded passages")
    parser.add_argument("--indexing-batch-size", type=int, default=1000000, help="Batch size of the number of passages indexed")
    parser.add_argument("--n-subquantizers", type=int, default=0, help="Number of subquantizers used for vector quantization, 0 for flat index")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits per subquantizer")

    args = parser.parse_args()
    main(args)
