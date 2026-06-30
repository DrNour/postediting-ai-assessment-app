from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class SegmentRecord(BaseModel):
    """One EduApp task record.

    source_text: Arabic source text.
    mt_output: English MT output shown to the user.
    user_submission: Human translation or post-edited output.
    """

    instance_id: str | None = None
    participant_id: str | None = None
    exercise_id: str | None = None
    task_type: Literal["post_editing", "translation", "unknown"] = "unknown"
    source_text: str = ""
    mt_output: str = ""
    user_submission: str = ""
    time_spent_sec: float | None = None
    preferred_edit_count: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MetricConfig(BaseModel):
    """Feature switches for the metrics layer.

    Heavy metrics are optional because they require model downloads and/or API calls.
    """

    use_sacrebleu: bool = True
    use_bertscore: bool = False
    use_comet: bool = False
    use_qe: bool = False
    use_embeddings: bool = False
    use_llm_judge: bool = False

    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    embedding_model: str = "sentence-transformers/LaBSE"
    comet_model: str = "Unbabel/wmt22-comet-da"
    qe_model: str = "Unbabel/wmt23-cometkiwi-da-xl"
    llm_model: str = "gpt-5.5"

    batch_size: int = 8
    device: str | None = None
    cache_dir: str | None = None


class MetricResult(BaseModel):
    instance_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MQMErrorSpan(BaseModel):
    span: str = ""
    category: Literal["accuracy", "fluency", "terminology", "style", "locale", "other"] = "other"
    severity: Literal["minor", "major", "critical"] = "minor"
    explanation: str = ""


class LLMJudgeResult(BaseModel):
    adequacy: int = Field(ge=0, le=100)
    fluency: int = Field(ge=0, le=100)
    terminology: int = Field(ge=0, le=100)
    style: int = Field(ge=0, le=100)
    overall_quality: int = Field(ge=0, le=100)
    estimated_postediting_effort: Literal["low", "medium", "high"]
    severity: Literal["none", "minor", "major", "critical"]
    error_spans: list[MQMErrorSpan] = Field(default_factory=list)
    brief_rationale: str = ""
