#!/bin/bash
# Step 0: Passage Retrieval
# Run from the project root directory.

# 0-1. Generate passage embeddings
python retrieval/generate_embeddings.py \
    --model_name_or_path <RETRIEVER_MODEL_PATH> \
    --output_dir <EMBEDDING_OUTPUT_DIR> \
    --passages <PASSAGES_TSV_PATH> \
    --shard_id 0 \
    --num_shards 1

# 0-2. Retrieve passages
python retrieval/retrieve_passages.py \
    --model_name_or_path <RETRIEVER_MODEL_PATH> \
    --passages <PASSAGES_TSV_PATH> \
    --passages_embeddings <EMBEDDING_DIR> \
    --data <INPUT_DATA_PATH> \
    --n_docs 5 \
    --output_dir <OUTPUT_DIR> \
    --projection_size 512
