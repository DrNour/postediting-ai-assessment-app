"""Human-authored MQM prompt templates for EduApp's adaptive condition."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


MQM_PROMPT_OPTIONS = (
    "Accuracy / meaning",
    "Terminology",
    "Fluency / style",
    "Register / audience",
    "Locale conventions",
)

_PROMPT_SPECS = {
    "Accuracy / meaning": {
        "mqm_category": "accuracy",
        "instruction": (
            "Identify a possible meaning, omission, addition, or ambiguity risk. "
            "Ask the learner to compare the relevant source and draft segment. "
            "Explain the risk, but do not provide a complete replacement translation."
        ),
    },
    "Terminology": {
        "mqm_category": "terminology",
        "instruction": (
            "Identify terms that may need a more consistent or domain-appropriate rendering. "
            "Offer concise alternatives with usage notes. Do not rewrite the full translation."
        ),
    },
    "Fluency / style": {
        "mqm_category": "fluency",
        "instruction": (
            "Identify local wording that may sound unnatural or conflict with target-language genre conventions. "
            "Explain why and suggest a local improvement, not a full rewrite."
        ),
    },
    "Register / audience": {
        "mqm_category": "style/register",
        "instruction": (
            "Assess whether tone, formality, institutional voice, and genre suit the intended audience. "
            "Ask a diagnostic question and offer a local alternative only where useful."
        ),
    },
    "Locale conventions": {
        "mqm_category": "locale convention",
        "instruction": (
            "Check dates, names, units, punctuation, cultural references, and other locale-sensitive conventions. "
            "Explain any mismatch and invite the learner to decide on an appropriate local revision."
        ),
    },
}


def get_mqm_prompt_spec(help_type: str) -> dict[str, str]:
    """Return a copy of the human-authored prompt specification for one choice."""
    selected = help_type if help_type in _PROMPT_SPECS else MQM_PROMPT_OPTIONS[0]
    return {"help_type": selected, **_PROMPT_SPECS[selected]}


def build_mqm_instruction(help_type: str) -> str:
    """Return the diagnostic instruction sent to the LLM for the selected MQM area."""
    return get_mqm_prompt_spec(help_type)["instruction"]


def prompt_event(
    *,
    help_type: str,
    student_question: str,
    draft_before_prompt: str,
    response_text: str | None = None,
    status: str = "requested",
) -> dict[str, Any]:
    """Create a minimised, auditable prompt-use record for later research export."""
    spec = get_mqm_prompt_spec(help_type)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "help_type": spec["help_type"],
        "mqm_category": spec["mqm_category"],
        "student_question": (student_question or "").strip(),
        "draft_word_count_before_prompt": len((draft_before_prompt or "").split()),
        "response_word_count": len((response_text or "").split()),
        "status": status,
    }
