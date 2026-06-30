from __future__ import annotations

import json
import os
from typing import Any

from .metrics import normalize_text
from .schemas import LLMJudgeResult, SegmentRecord


LLM_JUDGE_SYSTEM = """
You are an expert evaluator of Arabic-to-English machine translation and post-editing.
Judge the English MT output against the Arabic source. If a human post-edit is provided,
use it only as supporting evidence of what the human changed; do not assume it is perfect.
Return only valid JSON matching the supplied schema.
""".strip()


def build_judge_prompt(record: SegmentRecord) -> str:
    return f"""
Evaluate the Arabic-to-English machine translation record.

Arabic source:
{normalize_text(record.source_text)}

Machine translation output:
{normalize_text(record.mt_output)}

Human submission / post-edit, if available:
{normalize_text(record.user_submission)}

Scoring instructions:
- adequacy: 0 means the English misses or distorts the source; 100 means the meaning is fully preserved.
- fluency: 0 means unusable English; 100 means natural and grammatical English.
- terminology: 0 means serious lexical/term problems; 100 means appropriate terminology.
- style: 0 means inappropriate style/register; 100 means appropriate style/register.
- overall_quality: holistic MT quality from 0 to 100.
- estimated_postediting_effort: low, medium, or high.
- severity: none, minor, major, or critical.
- error_spans: short English spans from the MT output that need attention, with category, severity, and explanation.
Keep the rationale brief and evidence-based.
""".strip()


JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "adequacy": {"type": "integer", "minimum": 0, "maximum": 100},
        "fluency": {"type": "integer", "minimum": 0, "maximum": 100},
        "terminology": {"type": "integer", "minimum": 0, "maximum": 100},
        "style": {"type": "integer", "minimum": 0, "maximum": 100},
        "overall_quality": {"type": "integer", "minimum": 0, "maximum": 100},
        "estimated_postediting_effort": {"type": "string", "enum": ["low", "medium", "high"]},
        "severity": {"type": "string", "enum": ["none", "minor", "major", "critical"]},
        "error_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "span": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["accuracy", "fluency", "terminology", "style", "locale", "other"],
                    },
                    "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
                    "explanation": {"type": "string"},
                },
                "required": ["span", "category", "severity", "explanation"],
            },
        },
        "brief_rationale": {"type": "string"},
    },
    "required": [
        "adequacy",
        "fluency",
        "terminology",
        "style",
        "overall_quality",
        "estimated_postediting_effort",
        "severity",
        "error_spans",
        "brief_rationale",
    ],
}


def judge_record_with_openai(
    record: SegmentRecord,
    model: str = "gpt-5.5",
    api_key: str | None = None,
) -> LLMJudgeResult:
    """Return a structured LLM-as-judge result using the OpenAI Responses API.

    Requires:
        pip install openai
        export OPENAI_API_KEY=...
    """
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install the OpenAI Python package: pip install openai") from exc

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": LLM_JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_prompt(record)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "arabic_english_mt_judgment",
                "schema": JSON_SCHEMA,
                "strict": True,
            }
        },
    )
    data = json.loads(response.output_text)
    return LLMJudgeResult(**data)


def flatten_judge_result(result: LLMJudgeResult) -> dict[str, Any]:
    severity_map = {"none": 0, "minor": 1, "major": 2, "critical": 3}
    effort_map = {"low": 0, "medium": 1, "high": 2}
    return {
        "llm_adequacy": result.adequacy,
        "llm_fluency": result.fluency,
        "llm_terminology": result.terminology,
        "llm_style": result.style,
        "llm_overall_quality": result.overall_quality,
        "llm_estimated_effort_label": result.estimated_postediting_effort,
        "llm_estimated_effort_numeric": effort_map[result.estimated_postediting_effort],
        "llm_severity_label": result.severity,
        "llm_severity_numeric": severity_map[result.severity],
        "llm_error_span_count": len(result.error_spans),
        "llm_major_or_critical_error_count": sum(1 for e in result.error_spans if e.severity in {"major", "critical"}),
        "llm_brief_rationale": result.brief_rationale,
        "llm_error_spans_json": json.dumps([e.model_dump() for e in result.error_spans], ensure_ascii=False),
    }
