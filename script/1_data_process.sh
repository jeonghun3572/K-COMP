## Example of running the data processing script
python ../code/data_process/0_extract_entity.py \
    --input_path  \
    --output_path  \

python ../code/data_process/1_gpt_batch_summary.py \
    --input_path  \
    --output_path  \

python ../code/data_process/2_wiki_preprocess.py \
    --task "filter" \
    --threads 100 \
    --data_folder ./wiki

python ../code/data_process/3_short_desc.py \
    --wiki_dump ./wiki/enwiki-latest-pages-articles.xml.bz2 \
    --out_dir ./wiki \

python ../code/data_process/4_entity_tag.py \
    --desc_data_wiki ./short_descriptions_wiki.json \
    --desc_data_med ./short_descriptions_med.json \
    --input_path \
    --output_path \