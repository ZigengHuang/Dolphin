# Dolphin Reproducible Sample Package

This package provides a privacy-preserving workflow example for running the
Dolphin patient-education pipeline on public sample inputs. It is intended to
support workflow-level reproducibility under clinical privacy constraints.

The package includes:

- deidentified synthetic conversation examples;
- synthetic audio files paired with the sample conversations;
- a small anonymized site-specific knowledge-base example adapted from
  non-identifying education entries in the local knowledge-base workbook;
- prompt templates for reasoning-path and response generation;
- a sample model/pipeline configuration file;
- a runnable inference script that produces structured JSONL outputs.

The sample data are not part of the clinical study dataset and should not be
used to reproduce the numerical claims reported in the manuscript. The raw
clinical conversation/audio data and individual-level trial data remain subject
to institutional review and signed data access agreements.

The sample knowledge base excludes hospital contact details, administrative
workflow entries, patient-level information, and any identifiers. Entries were
translated and generalized for public demonstration.

## Directory Layout

```text
reproducible_sample_package/
  configs/sample_config.json
  knowledge_base/sample_knowledge_base.csv
  prompts/reasoning_path_prompt.txt
  prompts/response_prompt.txt
  sample_data/audio/*.wav
  sample_data/conversations/sample_cases.jsonl
  scripts/run_sample_pipeline.py
```

## Quick Start

From the `Dolphin-main` directory:

```bash
python reproducible_sample_package/scripts/run_sample_pipeline.py \
  --config reproducible_sample_package/configs/sample_config.json \
  --output reproducible_sample_package/outputs/sample_outputs.jsonl
```

The default mode uses a deterministic template responder so the workflow can be
executed without a private LLM key. The script performs the following steps:

1. loads sample conversation/audio pairs;
2. extracts simple audio descriptors from the paired WAV files;
3. retrieves relevant entries from the sample knowledge base;
4. builds a structured reasoning path;
5. generates a patient-education response;
6. writes JSONL outputs for inspection.

## Optional OpenAI-Compatible LLM Mode

To run the final response-generation step with a compatible base LLM endpoint,
set these environment variables and pass `--llm-provider openai-compatible`:

```bash
export DOLPHIN_SAMPLE_LLM_ENDPOINT="https://your-endpoint/v1/chat/completions"
export DOLPHIN_SAMPLE_LLM_MODEL="your-model-name"
export DOLPHIN_SAMPLE_LLM_API_KEY="your-api-key"

python reproducible_sample_package/scripts/run_sample_pipeline.py \
  --config reproducible_sample_package/configs/sample_config.json \
  --llm-provider openai-compatible \
  --output reproducible_sample_package/outputs/sample_outputs_llm.jsonl
```

The endpoint must accept the OpenAI-style `chat/completions` request schema.

## Reproducibility Scope

This package supports reproducibility of:

- input loading and deidentification-safe file structure;
- audio/text preprocessing flow;
- knowledge-base retrieval flow;
- reasoning-path prompt construction;
- response prompt construction and inference call shape;
- dependency-light execution on public sample inputs.

This package does not provide:

- raw clinical conversation/audio recordings;
- the site-specific production knowledge base;
- private model weights or API keys;
- the individual-level randomized-trial dataset.

Those resources contain sensitive clinical information or institution-specific
deployment components and are governed by the manuscript's Data Availability
statement.
