# Data Processing

## Summary Generation via Batch API

In the data processing pipeline, `generate_summary.py` prepares batch requests for generating summaries from the `passage` field. We use the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) with `gpt-4o-mini` to generate summaries. After the batch API completes, include the `summary` field directly in the data.

## Description Data

Medical description data is provided in `data/short_descriptions_med.json`.

For Wikipedia descriptions, we have included preprocessing code. For a detailed implementation, please refer to the [KILM repository](https://github.com/alexa/kilm/tree/main/data/wiki).

Before running `tag_entities.py`, the description data should be structured as:

```json
[
  {
    "title": "Entity Name",
    "description": "Short description of the entity."
  }
]
```
