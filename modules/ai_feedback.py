import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from modules.taxonomy import ERROR_TAXONOMY, RUBRIC

load_dotenv()


def get_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_client():
    api_key = get_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env or Streamlit secrets.")
    return OpenAI(api_key=api_key)


def build_feedback_prompt(
    source_text,
    mt_output,
    post_edited_text,
    editing_time_seconds=None,
    lexical_similarity=None,
    change_ratio=None,
    mt_pe_semantic_similarity=None,
    source_pe_semantic_similarity=None,
):
    return f"""
You are assisting a translation teacher.

Your task is to analyse a student's post-edited translation.
Use the taxonomy and rubric provided below.

SOURCE TEXT:
{source_text}

MACHINE TRANSLATION:
{mt_output}

STUDENT POST-EDITED VERSION:
{post_edited_text}

METRICS:
- Editing time in seconds: {editing_time_seconds}
- MT vs PE lexical similarity: {lexical_similarity}
- Change ratio: {change_ratio}
- MT vs PE semantic similarity: {mt_pe_semantic_similarity}
- Source vs PE semantic similarity: {source_pe_semantic_similarity}

ERROR TAXONOMY:
{json.dumps(ERROR_TAXONOMY, ensure_ascii=False, indent=2)}

RUBRIC:
{json.dumps(RUBRIC, ensure_ascii=False, indent=2)}

INSTRUCTIONS:
1. Identify possible translation or post-editing issues.
2. Classify each issue using the taxonomy.
3. Use only these main categories: accuracy, terminology, fluency, style_register, locale_cultural, formatting.
4. Assign severity: minor, major, or critical.
5. Provide evidence from the text.
6. Give student-friendly pedagogical feedback.
7. If uncertain, say teacher review is needed.
8. Do not invent problems that are not supported by the texts.
9. Return only valid JSON.

REQUIRED JSON FORMAT:
{{
  "overall_comment": "...",
  "possible_errors": [
    {{
      "category": "...",
      "subcategory": "...",
      "severity": "...",
      "evidence": "...",
      "explanation": "...",
      "suggested_revision": "...",
      "confidence": "high/medium/low"
    }}
  ],
  "rubric_scores": {{
    "accuracy": 0,
    "fluency": 0,
    "terminology": 0,
    "style_register": 0,
    "mechanics_formatting": 0
  }},
  "teacher_review_required": true
}}
"""


def generate_ai_feedback(
    source_text,
    mt_output,
    post_edited_text,
    editing_time_seconds=None,
    lexical_similarity=None,
    change_ratio=None,
    mt_pe_semantic_similarity=None,
    source_pe_semantic_similarity=None,
    model_name="gpt-4.1-mini",
):
    prompt = build_feedback_prompt(
        source_text=source_text,
        mt_output=mt_output,
        post_edited_text=post_edited_text,
        editing_time_seconds=editing_time_seconds,
        lexical_similarity=lexical_similarity,
        change_ratio=change_ratio,
        mt_pe_semantic_similarity=mt_pe_semantic_similarity,
        source_pe_semantic_similarity=source_pe_semantic_similarity,
    )

    client = get_client()
    response = client.responses.create(model=model_name, input=prompt)
    raw_text = response.output_text

    try:
        feedback_json = json.loads(raw_text)
        return {"success": True, "feedback": feedback_json, "raw_output": raw_text}
    except json.JSONDecodeError:
        return {
            "success": False,
            "feedback": None,
            "raw_output": raw_text,
            "error": "The model did not return valid JSON.",
        }


def check_feedback_risk(feedback):
    warnings = []

    if not feedback:
        return {"risk_level": "high", "warnings": ["No valid AI feedback was generated."]}

    if "possible_errors" not in feedback:
        warnings.append("AI output does not include possible_errors.")
    if "overall_comment" not in feedback:
        warnings.append("AI output does not include an overall_comment.")

    for error in feedback.get("possible_errors", []):
        if not error.get("evidence"):
            warnings.append("One or more errors have no textual evidence.")
        if error.get("confidence") == "low":
            warnings.append("One or more AI comments have low confidence.")
        if error.get("category") not in ERROR_TAXONOMY:
            warnings.append("AI used a category outside the approved taxonomy.")

    if len(warnings) == 0:
        return {"risk_level": "low", "warnings": []}
    if len(warnings) <= 2:
        return {"risk_level": "medium", "warnings": warnings}
    return {"risk_level": "high", "warnings": warnings}
