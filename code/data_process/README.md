## Generate summary via batch API
In `1_data_process.sh`, `1_gpt_batch_summary.py` is a preparatory step for generating a summary from the `passage` field. We use the OpenAI batch api to have the gpt-4o-mini model generate a summary. After the batch api, please include the `summary` directly in the data. For more information about the batch API, please see [[Batch API](https://platform.openai.com/docs/guides/batch)].

## Description data
Medical description data can be found in the file. `short_descriptions_med.json`

We have included the code for the `wiki description`, but please refer to the corresponding [github](https://github.com/alexa/kilm/tree/main/data/wiki) for a detailed implementation.

Before running `4_entity_tag.py`, the description should be structured as follows.
```
{
  "title": "",
  "description": ""
}
```