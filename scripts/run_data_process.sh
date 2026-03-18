#!/bin/bash
# Step 1: Data Processing Pipeline
# Run from the project root directory.
# Execute each step sequentially after verifying the output of the previous step.

# 1-1. Extract entities from questions
python data_processing/extract_entity.py \
    --input_path <RETRIEVED_DATA_PATH> \
    --output_path <ENTITY_OUTPUT_PATH>

# 1-2. Generate GPT batch summary requests
python data_processing/generate_summary.py \
    --input_path <ENTITY_OUTPUT_PATH> \
    --output_path <BATCH_REQUEST_PATH>

# 1-3. Preprocess Wikipedia data
python data_processing/preprocess_wiki.py \
    --task "filter" \
    --threads 100 \
    --data_folder <WIKI_DATA_DIR>

# 1-4. Extract short descriptions from Wikipedia dump
python data_processing/extract_short_desc.py \
    --wiki_dump <WIKI_DUMP_PATH> \
    --out_dir <WIKI_OUTPUT_DIR>

# 1-5. Tag entities with descriptions
python data_processing/tag_entities.py \
    --desc_data_wiki <WIKI_DESC_PATH> \
    --desc_data_med data/short_descriptions_med.json \
    --input_path <ENTITY_OUTPUT_PATH> \
    --output_path <TAGGED_OUTPUT_PATH>

# 1-6. Apply question masking and create final dataset
python data_processing/mask_questions.py \
    <TAGGED_OUTPUT_PATH> \
    <FINAL_DATASET_PATH>
