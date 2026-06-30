from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from io import StringIO, BytesIO

from .llm_judge import flatten_judge_result, judge_record_with_openai
from .metrics import compute_metrics_for_dataframe, compute_metrics_for_record
from .schemas import MetricConfig, SegmentRecord

router = APIRouter(tags=["EduApp metrics"])


@router.post("/metrics/segment")
def metrics_for_segment(record: SegmentRecord, config: MetricConfig | None = None):
    """Compute lightweight metrics for one EduApp segment.

    For COMET, BERTScore, embeddings, or batch efficiency, use /metrics/upload-csv.
    """
    cfg = config or MetricConfig(use_bertscore=False, use_comet=False, use_qe=False, use_embeddings=False)
    return compute_metrics_for_record(record, cfg).model_dump()


@router.post("/metrics/llm-judge")
def llm_judge_for_segment(record: SegmentRecord, model: str = "gpt-5.5"):
    """Run optional LLM-as-judge scoring for one segment."""
    try:
        result = judge_record_with_openai(record, model=model)
        return flatten_judge_result(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/metrics/upload-csv")
async def metrics_for_csv(
    file: UploadFile = File(...),
    use_sacrebleu: bool = True,
    use_bertscore: bool = False,
    use_comet: bool = False,
    use_qe: bool = False,
    use_embeddings: bool = False,
):
    """Upload a CSV and receive an enriched CSV with metrics added."""
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))
        cfg = MetricConfig(
            use_sacrebleu=use_sacrebleu,
            use_bertscore=use_bertscore,
            use_comet=use_comet,
            use_qe=use_qe,
            use_embeddings=use_embeddings,
        )
        enriched = compute_metrics_for_dataframe(df, cfg)
        buf = StringIO()
        enriched.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=eduapp_metrics_enriched.csv"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
