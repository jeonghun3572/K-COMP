## Example of running the passage retrieval script
export PYTHONPATH=../code

python ../code/retrieval/generate_passage_embeddings.py \
    --model-name-or-path  \
    --passages  \
    --output-dir  \
    --shard-id 0 \
    --num-shards 1 \
    --projection-size 512

python ../code/retrieval/passage_retrieval.py \
    --model-name-or-path  \
    --data  \
    --passages  \
    --passages-embeddings  \
    --output-dir  \
    --n-docs 5 \
    --projection-size 512
