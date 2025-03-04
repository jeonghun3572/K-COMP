## Example of running the passage retrieval script

python 0_generate_passage_embeddings.py \
    --model_name_or_path  \
    --output_dir   \
    --passages  \
    --shard_id 0 --num_shards 1 \

python 1_passage_retrieval.py \
    --model_name_or_path  \
    --passages  \
    --passages_embeddings  \
    --data  \
    --n_docs 5 \
    --output_dir  \
    --projection_size 512 \