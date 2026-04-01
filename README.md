<h1 align="center">K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2501.13567"><img src="https://img.shields.io/badge/arXiv-2501.13567-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/jeonghuncho/K-COMP"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg" alt="HuggingFace"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

<p align="center"><img src="assets/overview.png" width="70%"></p>
<p align="center"><i>Overview of the K-COMP framework.</i></p>

K-COMP is a retrieval-augmented generation (RAG) framework for medical question answering. It improves answer quality by compressing retrieved passages into entity-aware summaries using a fine-tuned language model that has been injected with entity-level knowledge.

## Project Structure

```
K-COMP/
├── train.py                 # Model training
├── inference.py             # Model inference
├── src/                     # Core library
│   ├── data.py              # Dataset & collator
│   ├── index.py             # FAISS indexer
│   ├── normalize_text.py    # Unicode normalization
│   ├── slurm.py             # Distributed / SLURM utilities
│   └── util.py              # Wikipedia processing utilities
├── retrieval/               # Passage retrieval pipeline
│   ├── generate_embeddings.py
│   └── retrieve_passages.py
├── data_processing/         # Data preprocessing pipeline
│   ├── README.md            # Detailed pipeline notes
│   ├── extract_entity.py    # Step 1-1: Extract entities from questions
│   ├── generate_summary.py  # Step 1-2: Prepare GPT batch requests
│   ├── preprocess_wiki.py   # Step 1-3: Filter Wikipedia articles
│   ├── extract_short_desc.py# Step 1-4: Extract Wikipedia short descriptions
│   ├── tag_entities.py      # Step 1-5: Tag entities with descriptions
│   └── mask_questions.py    # Step 1-6: Mask entities & build training data
├── scripts/                 # Shell scripts for each stage
│   ├── run_retrieval.sh     # Step 0
│   ├── run_data_process.sh  # Step 1
│   ├── run_train.sh         # Step 2
│   └── run_inference.sh     # Step 3
└── data/
    └── short_descriptions_med.json  # Medical entity descriptions
```

## Prerequisites

- Python 3.10+
- CUDA 11.8+ with compatible GPU drivers
- GPU with ≥24 GB VRAM (training) / ≥8 GB VRAM (inference)
- [OpenAI API key](https://platform.openai.com/api-keys) — required for data processing Step 1-2

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies

```bash
uv venv --python 3.10
source .venv/bin/activate

uv sync
```

### 3. Install flash-attn

`flash-attn` requires CUDA build tools and must be installed separately:

```bash
uv pip install flash-attn --no-build-isolation
```

### 4. Install the scispaCy model

The entity extractor requires a scientific NLP model:

```bash
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
```

## Datasets

Download the QA datasets and place them in your preferred data directory:

- [MedQuAD](https://github.com/abachaa/MedQuAD) — Question answering via question entailment
- [MASH-QA](https://github.com/mingzhu0527/MASHQA) — Long multiple-span QA
- [BioASQ](https://participants-area.bioasq.org/datasets/) — Biomedical QA

Download the retrieval corpus:

- [MedCorp](https://github.com/Teddy-XiongGZ/MedRAG) — Medical RAG benchmark corpus

## Data Format

Each stage of the pipeline expects and produces specific JSON structures.

### Input QA data (before retrieval)

```jsonc
// JSONL — one record per line
{"question": "What are the symptoms of diabetes?", "answer": "Common symptoms include..."}
```

### After Step 0 — Retrieval

```jsonc
// JSONL
{
  "question": "What are the symptoms of diabetes?",
  "answer": "Common symptoms include...",
  "ctxs": [
    {"id": "1234", "title": "Diabetes mellitus", "text": "Diabetes mellitus is a metabolic disease..."},
    ...
  ]
}
```

### After Step 1 — Data processing

```jsonc
// JSON — training-ready format
{
  "prompt": "### Question\nWhat are the symptoms of <ent>?\n\n### Passage\nDiabetes mellitus\nDiabetes mellitus is...",
  "completion": "### Entity\ndiabetes: a chronic metabolic disease...<eod>\n\n### Summary\nDiabetes is characterized by..."
}
```

> The `<ent>` token marks masked entities in the question, and `<eod>` separates entity descriptions.
> See [`data_processing/README.md`](data_processing/README.md) for the full intermediate formats and the manual step required between Step 1-2 and 1-3.

## Usage

All scripts must be run from the **project root directory**.

### Step 0 — Passage Retrieval

```bash
bash scripts/run_retrieval.sh
```

Fill in the `<PLACEHOLDER>` values in `scripts/run_retrieval.sh` before running.

### Step 1 — Data Processing

```bash
bash scripts/run_data_process.sh
```

> **Manual step required between 1-2 and 1-3.**
> `generate_summary.py` creates a batch request file for the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch). You must submit this file, wait for completion, then add the returned `summary` field back into your data before running steps 1-3 onward.
> See [`data_processing/README.md`](data_processing/README.md) for details.

### Step 2 — Training

```bash
bash scripts/run_train.sh
```

Fill in the `<PLACEHOLDER>` values in `scripts/run_train.sh` (model path, data paths, output directory, etc.).

### Step 3 — Inference

```bash
bash scripts/run_inference.sh
```

Results are saved to `result/<DATASET_NAME>/summ/<LOG_NAME>.jsonl`.

## Model Checkpoints

Pretrained checkpoints are available on [Hugging Face](https://huggingface.co/jeonghuncho/K-COMP).

## Citation

```bibtex
@misc{cho2025kcompretrievalaugmentedmedicaldomain,
      title={K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor},
      author={Jeonghun Cho and Gary Geunbae Lee},
      year={2025},
      eprint={2501.13567},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.13567},
}
```

## Acknowledgements

This repository builds upon:

- [Contriever](https://github.com/facebookresearch/contriever) — Unsupervised dense information retrieval
- [KILM](https://github.com/alexa/kilm) — Knowledge injection into encoder-decoder language models
- [TRL](https://github.com/huggingface/trl) — Transformer reinforcement learning library

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
