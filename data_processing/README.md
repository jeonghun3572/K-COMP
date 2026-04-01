# Data Processing Pipeline

The pipeline runs in 6 sequential steps. Each step's output becomes the next step's input.

## Overview

```
Input QA data (JSONL)
    │
    ▼
[1-1] extract_entity.py     — extract medical entities from questions
    │
    ▼
[1-2] generate_summary.py   — build OpenAI Batch API request file
    │
    ▼  ← ⚠️  MANUAL STEP: submit batch, wait, merge results (see below)
    │
[1-3] preprocess_wiki.py    — filter & clean Wikipedia dump
[1-4] extract_short_desc.py — extract short descriptions from Wikipedia
    │
    ▼
[1-5] tag_entities.py       — match entities with descriptions
    │
    ▼
[1-6] mask_questions.py     — mask entities, build prompt/completion pairs
    │
    ▼
Training data (JSON)
```

---

## Step 1-1: Entity Extraction

**Script:** `extract_entity.py`

**Input:** JSONL file (output of retrieval step)

```jsonc
{"question": "...", "answer": "...", "ctxs": [{"id": "...", "title": "...", "text": "..."}, ...]}
```

**Output:** JSON file

```jsonc
{
  "question": "...",
  "answer": "...",
  "passage": "Title1\ntext1\n\nTitle2\ntext2",
  "entity": ["diabetes", "insulin"],
  "ctxs": [...]
}
```

---

## Step 1-2: Summary Request Generation

**Script:** `generate_summary.py`

Generates a JSONL batch request file for the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) using `gpt-4o-mini`. Each request asks the model to extract a passage summary (≤4 sentences) for the given question.

**Input:** JSON file from Step 1-1

**Output:** JSONL batch request file (ready to upload to OpenAI)

---

## ⚠️ Manual Step: Submit Batch and Merge Results

After Step 1-2, you must:

1. **Upload** the batch request file to the OpenAI Batch API:

```python
from openai import OpenAI
client = OpenAI()

with open("batch_requests.jsonl", "rb") as f:
    batch_file = client.files.create(file=f, purpose="batch")

batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
print(batch.id)  # save this ID
```

2. **Wait** for the batch to complete (check with `client.batches.retrieve(batch_id)`).

3. **Download** the results and **merge** the `summary` field back into your data. Each record should then contain:

```jsonc
{
  "question": "...",
  "answer": "...",
  "passage": "...",
  "entity": [...],
  "ctxs": [...],
  "summary": "Diabetes is a chronic metabolic disease characterized by..."  // ← add this
}
```

Only after merging the `summary` field should you proceed to Steps 1-3 onward.

---

## Steps 1-3 & 1-4: Wikipedia Processing

**Scripts:** `preprocess_wiki.py`, `extract_short_desc.py`

These scripts process a Wikipedia XML dump to extract short entity descriptions.

- `preprocess_wiki.py` — filters and cleans Wikipedia articles from a pre-downloaded dump
- `extract_short_desc.py` — extracts short descriptions from the dump using an SAX parser

For a detailed reference implementation, see the [KILM repository](https://github.com/alexa/kilm/tree/main/data/wiki).

**Output format** (required by Step 1-5):

```json
[
  {"title": "Entity Name", "description": "Short description of the entity."},
  ...
]
```

Medical descriptions are already provided in `data/short_descriptions_med.json`.

---

## Step 1-5: Entity Tagging

**Script:** `tag_entities.py`

Matches each extracted entity against the Wikipedia and medical description databases and attaches a formatted description string.

> **Note:** This script uses `parmap` with `num_cores = 55` for parallelism. Adjust this value in `tag_entities.py` to match the number of CPU cores on your machine.

**Output:** JSON with added `description` field:

```jsonc
{
  ...,
  "description": "diabetes: a chronic metabolic disease...<eod>\ninsulin: a peptide hormone...<eod>"
}
```

---

## Step 1-6: Question Masking

**Script:** `mask_questions.py`

Masks entity mentions in the question with `<ent>` tokens and produces the final `prompt`/`completion` pairs for supervised fine-tuning.

**Output:** JSON — training-ready format:

```jsonc
{
  "prompt": "### Question\nWhat are the symptoms of <ent>?\n\n### Passage\nDiabetes mellitus\n...",
  "completion": "### Entity\ndiabetes: a chronic metabolic disease...<eod>\n\n### Summary\nDiabetes is characterized by..."
}
```
