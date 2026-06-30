# EduApp-PE IP&M Upgrade: Metrics, QE, and LLM-as-Judge Layer

This package adds a state-of-the-art evaluation layer to EduApp-PE so the manuscript can move beyond length/telemetry/edit-distance baselines and include neural MT evaluation, quality estimation, embedding features, and optional LLM-as-judge diagnostics.

## What this adds

1. Traditional post-editing effort metrics
   - Levenshtein edit distance
   - normalized edit ratio
   - word/character counts
   - time-normalized productivity indicators

2. Reference-based MT evaluation using the human post-edit as a reference
   - BLEU
   - chrF / chrF++
   - TER
   - BERTScore
   - COMET reference-based scoring

3. Reference-free quality estimation
   - COMETKiwi / QE scores using source + MT only

4. Multilingual semantic features
   - source/MT, source/post-edit, and MT/post-edit embedding similarities

5. LLM-as-judge features
   - adequacy, fluency, terminology, style, severity, post-editing need, and MQM-style error spans
   - structured JSON output suitable for reproducible downstream modeling

6. New modeling layer
   - F5 neural metric features
   - F6 embedding features
   - F7 LLM judge features
   - F8 combined SOTA feature set
   - participant-grouped cross-validation

## Install

```bash
cd eduapp_ipm_upgrade_code
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

COMET and BERTScore may download model weights on first use. Use a machine with sufficient RAM/GPU if possible.

## Data format

Your CSV should contain, at minimum:

```text
instance_id,participant_id,exercise_id,task_type,source_text,mt_output,user_submission,time_spent_sec,preferred_edit_count
```

The code is defensive: if some columns are missing, it will compute what it can and leave optional metrics blank.

## Batch enrichment

```bash
python run_batch_metrics.py \
  --input path/to/post_editing.csv \
  --output path/to/post_editing_enriched.csv \
  --use_sacrebleu \
  --use_embeddings
```

To add COMET and COMETKiwi:

```bash
python run_batch_metrics.py \
  --input path/to/post_editing.csv \
  --output path/to/post_editing_enriched.csv \
  --use_sacrebleu \
  --use_bertscore \
  --use_comet \
  --use_qe \
  --use_embeddings
```

To add LLM-as-judge features, set `OPENAI_API_KEY` first:

```bash
export OPENAI_API_KEY="your-key"
python run_batch_metrics.py \
  --input path/to/post_editing.csv \
  --output path/to/post_editing_enriched.csv \
  --use_llm_judge
```

## Run grouped-CV models

```bash
python modeling_extension.py \
  --input path/to/post_editing_enriched.csv \
  --target preferred_edit_count \
  --group participant_id \
  --output_dir results
```

This writes regression and classification tables that can be pasted into the revised manuscript.

## FastAPI router

Add this router to your existing API:

```python
from fastapi import FastAPI
from eduapp_metrics.api_router import router as metrics_router

app = FastAPI()
app.include_router(metrics_router, prefix="/api")
```

Then run:

```bash
uvicorn your_api_file:app --reload
```

## Streamlit integration

See `streamlit_integration_snippet.py` for a minimal example that computes metrics at submission time and stores them with the report.

## Important research note

Do not present LLM judgments as human quality labels. In the manuscript, treat them as automated features or weak supervision, then validate them against human post-editing traces and, ideally, a human-rated subset.
