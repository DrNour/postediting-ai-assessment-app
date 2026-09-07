"""Shared helpers for translation and post-editing task modes."""

from __future__ import annotations

from typing import Any, MutableMapping

TRANSLATION = "translation"
ADAPTIVE_TRANSLATION = "adaptive_translation"
POST_EDITING = "post_editing"

TASK_OPTIONS = {
    "Translate without AI": TRANSLATION,
    "Adaptive translation with AI": ADAPTIVE_TRANSLATION,
    "Post-edit the machine translation": POST_EDITING,
}

# These values describe editing effort. They are not meaningful when a student
# translates independently from the source text, whether unaided or with adaptive AI support.
POST_EDITING_ONLY_METRIC_FIELDS = (
    "mt_pe_word_count_difference",
    "mt_pe_cosine_similarity",
    "mt_pe_edit_distance_ratio",
    "mt_pe_length_ratio",
    "mt_pe_lexical_similarity",
    "mt_pe_change_ratio",
    "mt_pe_inserted_words",
    "mt_pe_deleted_words",
    "mt_pe_replaced_words",
    "mt_pe_replacement_output_words",
    "mt_pe_unchanged_words",
    "mt_pe_changed_original_words",
    "mt_pe_unchanged_ratio",
    "mt_pe_changed_ratio_original",
    "mt_pe_overlap_bleu",
    "mt_pe_overlap_chrf",
    "mt_pe_overlap_ter",
    "mt_pe_bleu",
    "mt_pe_chrf",
    "mt_pe_ter",
    "mt_pe_bertscore_f1",
)


def normalize_task_type(value: Any) -> str:
    """Return a stable task identifier and preserve compatibility with old rows."""
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {TRANSLATION, "translate", "human_translation", "translation_without_ai"}:
        return TRANSLATION
    if normalized in {
        ADAPTIVE_TRANSLATION,
        "adaptive",
        "ai_translation",
        "ai_assisted_translation",
        "adaptive_ai_translation",
    }:
        return ADAPTIVE_TRANSLATION
    return POST_EDITING


def is_translation(value: Any) -> bool:
    return normalize_task_type(value) in {TRANSLATION, ADAPTIVE_TRANSLATION}


def is_adaptive_translation(value: Any) -> bool:
    return normalize_task_type(value) == ADAPTIVE_TRANSLATION


def task_type_label(value: Any) -> str:
    normalized = normalize_task_type(value)
    if normalized == ADAPTIVE_TRANSLATION:
        return "Adaptive Translation (AI-assisted)"
    if normalized == TRANSLATION:
        return "Translation (No AI)"
    return "Post-editing"


def student_output_label(value: Any) -> str:
    return "Student Translation" if is_translation(value) else "Post-Edited Text"


def task_instruction(value: Any) -> str:
    normalized = normalize_task_type(value)
    if normalized == TRANSLATION:
        return (
            "Translate the source text independently without AI assistance. The machine translation "
            "is hidden so that it does not influence your wording."
        )
    if normalized == ADAPTIVE_TRANSLATION:
        return (
            "Translate the source text with the built-in adaptive AI assistant. Ask for terminology, "
            "meaning, draft review, or next-segment guidance when you need it. You remain responsible "
            "for the final translation and type the final wording yourself."
        )
    return (
        "Revise the raw machine translation. Correct meaning, terminology, grammar, "
        "style, and fluency while preserving the source message."
    )


def make_translation_metrics_task_appropriate(
    results: MutableMapping[str, Any],
    *,
    has_reference: bool,
) -> MutableMapping[str, Any]:
    """Remove misleading post-editing-effort values from either translation condition."""
    for field in POST_EDITING_ONLY_METRIC_FIELDS:
        results[field] = None

    if has_reference:
        results["mt_pe_interpretation"] = (
            "Translation task: post-editing effort metrics are not applicable. "
            "Reference-based quality metrics compare the student's translation with "
            "the independent reference translation."
        )
    else:
        results["mt_pe_interpretation"] = (
            "Translation task: post-editing effort metrics are not applicable. "
            "No independent reference translation was provided, so automatic quality "
            "metrics are limited."
        )

    return results
