## Example of running the data processing script
export PYTHONPATH=../code

python ../code/data_process/extract_entity.py \
    --input-path  \
    --output-path

# Run the generated requests with the OpenAI Batch API and add `summary` to the data
python ../code/data_process/gpt_batch_summary.py \
    --input-path  \
    --output-path

# short_desc.py must run before wiki_preprocess.py
python ../code/data_process/short_desc.py \
    --wiki-dump ./wiki/enwiki-latest-pages-articles.xml.bz2 \
    --out-dir ./wiki

python ../code/data_process/wiki_preprocess.py \
    --threads 100 \
    --data-folder ./wiki

python ../code/data_process/entity_tag.py \
    --desc-data-wiki ./wiki/short_descriptions_wiki.json \
    --desc-data-med ../short_descriptions_med.json \
    --input-path  \
    --output-path

python ../code/data_process/question_masking.py \
    --input-path  \
    --output-path
