"""
metrics.py

Research-oriented MT / HT / PE evaluation utilities for EduApp.

This file calculates:

1. MT-to-PE post-editing effort metrics
   - cosine similarity
   - edit-distance ratio
   - length ratio
   - lexical similarity
   - change ratio
   - inserted / deleted / replaced / unchanged words

2. Reference-based translation quality metrics
   - BLEU
   - chrF
   - TER
   - optional BERTScore
   - optional COMET through an external scorer

3. Teacher evaluation fields
   - teacher score
   - teacher feedback

Important:
- MT-to-PE comparison measures editing effort, not true translation quality.
- True quality metrics require an independent reference translation.
"""

import math
import re
from collections import Counter
from difflib import SequenceMatcher

from rapidfuzz import fuzz


# ============================================================
# Basic text utilities
# ============================================================

def safe_text(text):
    """
    Converts None or non-string values into safe strings.
    """
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def tokenize_words(text):
    """
    Unicode-aware word tokenizer.

    Works for English and Arabic better than simple split(),
    though it is still intentionally lightweight.
    """
    text = safe_text(text)
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def word_count(text):
    """
    Simple word count.
    """
    return len(tokenize_words(text))


def word_count_difference(original_text, revised_text):
    """
    Difference in word count from original to revised.
    """
    return word_count(revised_text) - word_count(original_text)


# ============================================================
# Similarity and change metrics
# ============================================================

def cosine_similarity(text_a, text_b):
    """
    Lightweight token-frequency cosine similarity.

    Returns a value between 0 and 1.
    Higher = more similar.
    Lower = more changed.
    """
    tokens_a = tokenize_words(text_a)
    tokens_b = tokenize_words(text_b)

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)

    vocabulary = set(counts_a) | set(counts_b)

    dot_product = sum(counts_a[token] * counts_b[token] for token in vocabulary)
    norm_a = math.sqrt(sum(counts_a[token] ** 2 for token in vocabulary))
    norm_b = math.sqrt(sum(counts_b[token] ** 2 for token in vocabulary))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def lexical_similarity(text_a, text_b):
    """
    RapidFuzz lexical similarity between two strings.

    Returns a value between 0 and 1.
    """
    text_a = safe_text(text_a)
    text_b = safe_text(text_b)

    if not text_a and not text_b:
        return 1.0

    return fuzz.ratio(text_a, text_b) / 100


def change_ratio(text_a, text_b):
    """
    Approximate lexical change ratio.

    Returns a value between 0 and 1.
    Higher = more changed.
    """
    return 1 - lexical_similarity(text_a, text_b)


def edit_distance_ratio(text_a, text_b):
    """
    Normalized edit-distance ratio based on SequenceMatcher.

    Returns a value between 0 and 1.
    Higher = more editing/change.
    """
    text_a = safe_text(text_a)
    text_b = safe_text(text_b)

    if not text_a and not text_b:
        return 0.0

    similarity = SequenceMatcher(None, text_a, text_b).ratio()
    return 1 - similarity


def length_ratio(revised_text, original_text):
    """
    Length ratio = revised word count / original word count.

    Example:
    - 1.00 = same length
    - 1.20 = revised text is 20% longer
    - 0.80 = revised text is 20% shorter
    """
    revised_count = word_count(revised_text)
    original_count = word_count(original_text)

    if original_count == 0:
        return None

    return revised_count / original_count


# ============================================================
# Word-level edit operations
# ============================================================

def word_edit_operations(original_text, revised_text):
    """
    Counts inserted, deleted, replaced, and unchanged words.

    Usually:
    - original_text = raw MT
    - revised_text = post-edited MT
    """

    original_tokens = tokenize_words(original_text)
    revised_tokens = tokenize_words(revised_text)

    matcher = SequenceMatcher(None, original_tokens, revised_tokens)

    inserted_words = 0
    deleted_words = 0
    replaced_words = 0
    replacement_output_words = 0
    unchanged_words = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        original_span = i2 - i1
        revised_span = j2 - j1

        if tag == "equal":
            unchanged_words += original_span

        elif tag == "insert":
            inserted_words += revised_span

        elif tag == "delete":
            deleted_words += original_span

        elif tag == "replace":
            replaced_words += original_span
            replacement_output_words += revised_span

    original_count = len(original_tokens)
    revised_count = len(revised_tokens)

    changed_original_words = deleted_words + replaced_words

    if original_count > 0:
        unchanged_ratio = unchanged_words / original_count
        changed_ratio_original = changed_original_words / original_count
    else:
        unchanged_ratio = None
        changed_ratio_original = None

    return {
        "inserted_words": inserted_words,
        "deleted_words": deleted_words,
        "replaced_words": replaced_words,
        "replacement_output_words": replacement_output_words,
        "unchanged_words": unchanged_words,
        "original_word_count": original_count,
        "revised_word_count": revised_count,
        "changed_original_words": changed_original_words,
        "unchanged_ratio": unchanged_ratio,
        "changed_ratio_original": changed_ratio_original,
    }


# ============================================================
# Reference-based metrics: BLEU, chrF, TER
# ============================================================

def reference_based_scores(candidate_text, reference_text):
    """
    Calculates BLEU, chrF, and TER using sacrebleu.

    candidate_text:
        MT, HT, or PE translation.

    reference_text:
        Independent reference translation.

    Returns:
        dict with bleu, chrf, ter.

    If sacrebleu is not installed, returns None values.
    """

    candidate_text = safe_text(candidate_text)
    reference_text = safe_text(reference_text)

    if not candidate_text or not reference_text:
        return {
            "bleu": None,
            "chrf": None,
            "ter": None,
        }

    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(
            [candidate_text],
            [[reference_text]],
        ).score

        chrf = sacrebleu.corpus_chrf(
            [candidate_text],
            [[reference_text]],
        ).score

        ter = sacrebleu.corpus_ter(
            [candidate_text],
            [[reference_text]],
        ).score

        return {
            "bleu": bleu,
            "chrf": chrf,
            "ter": ter,
        }

    except Exception:
        return {
            "bleu": None,
            "chrf": None,
            "ter": None,
        }


# ============================================================
# Optional BERTScore
# ============================================================

def bert_score(candidate_text, reference_text, language="en"):
    """
    Optional BERTScore F1.

    Requires:
        pip install bert-score

    If bert-score is not installed or fails, returns None.
    """

    candidate_text = safe_text(candidate_text)
    reference_text = safe_text(reference_text)

    if not candidate_text or not reference_text:
        return None

    try:
        from bert_score import score

        _, _, f1 = score(
            [candidate_text],
            [reference_text],
            lang=language,
            verbose=False,
        )

        return float(f1[0])

    except Exception:
        return None


# ============================================================
# Add quality metrics against an independent reference
# ============================================================

def add_reference_based_quality_scores(
    results,
    prefix,
    candidate_text,
    reference_text,
    source_text=None,
    use_bert=False,
    bert_language="en",
    comet_scorer=None,
):
    """
    Adds BLEU, chrF, TER, optional BERTScore, and optional COMET.

    These are quality-oriented only when reference_text is independent.
    """

    candidate_text = safe_text(candidate_text)
    reference_text = safe_text(reference_text)
    source_text = safe_text(source_text)

    if not candidate_text or not reference_text:
        results[f"{prefix}_bleu"] = None
        results[f"{prefix}_chrf"] = None
        results[f"{prefix}_ter"] = None
        results[f"{prefix}_bertscore_f1"] = None
        results[f"{prefix}_comet"] = None
        return results

    scores = reference_based_scores(
        candidate_text=candidate_text,
        reference_text=reference_text,
    )

    results[f"{prefix}_bleu"] = scores.get("bleu")
    results[f"{prefix}_chrf"] = scores.get("chrf")
    results[f"{prefix}_ter"] = scores.get("ter")

    if use_bert:
        results[f"{prefix}_bertscore_f1"] = bert_score(
            candidate_text,
            reference_text,
            language=bert_language,
        )
    else:
        results[f"{prefix}_bertscore_f1"] = None

    if comet_scorer is not None:
        try:
            results[f"{prefix}_comet"] = comet_scorer(
                source=source_text,
                candidate=candidate_text,
                reference=reference_text,
            )
        except TypeError:
            try:
                results[f"{prefix}_comet"] = comet_scorer(
                    source_text,
                    candidate_text,
                    reference_text,
                )
            except Exception:
                results[f"{prefix}_comet"] = None
        except Exception:
            results[f"{prefix}_comet"] = None
    else:
        results[f"{prefix}_comet"] = None

    return results


# ============================================================
# Main full comparison function
# ============================================================

def compare_postedit_with_raw_mt(
    raw_mt,
    post_edited_text,
    human_translation=None,
    reference_text=None,
    source_text=None,
    teacher_score=None,
    teacher_feedback=None,
    use_bert=False,
    bert_language="en",
    comet_scorer=None,
):
    """
    Evaluates raw MT, human translation, and post-edited MT.

    Layers:
    1. MT-to-PE effort
    2. Reference-based quality
    3. Teacher assessment
    """

    raw_mt = safe_text(raw_mt)
    post_edited_text = safe_text(post_edited_text)
    human_translation = safe_text(human_translation)
    reference_text = safe_text(reference_text)
    source_text = safe_text(source_text)
    teacher_feedback = safe_text(teacher_feedback)

    results = {}

    # --------------------------------------------------------
    # 1. Word counts
    # --------------------------------------------------------

    results["raw_mt_word_count"] = word_count(raw_mt)
    results["pe_word_count"] = word_count(post_edited_text)
    results["mt_pe_word_count_difference"] = word_count_difference(
        raw_mt,
        post_edited_text,
    )

    results["ht_word_count"] = (
        word_count(human_translation) if human_translation else None
    )

    results["reference_word_count"] = (
        word_count(reference_text) if reference_text else None
    )

    # --------------------------------------------------------
    # 2. MT-to-PE post-editing effort metrics
    # --------------------------------------------------------

    results["mt_pe_cosine_similarity"] = cosine_similarity(
        raw_mt,
        post_edited_text,
    )

    results["mt_pe_edit_distance_ratio"] = edit_distance_ratio(
        raw_mt,
        post_edited_text,
    )

    results["mt_pe_length_ratio"] = length_ratio(
        post_edited_text,
        raw_mt,
    )

    results["mt_pe_lexical_similarity"] = round(
        lexical_similarity(raw_mt, post_edited_text),
        3,
    )

    results["mt_pe_change_ratio"] = round(
        change_ratio(raw_mt, post_edited_text),
        3,
    )

    edit_ops = word_edit_operations(
        raw_mt,
        post_edited_text,
    )

    results["mt_pe_inserted_words"] = edit_ops["inserted_words"]
    results["mt_pe_deleted_words"] = edit_ops["deleted_words"]
    results["mt_pe_replaced_words"] = edit_ops["replaced_words"]
    results["mt_pe_replacement_output_words"] = edit_ops["replacement_output_words"]
    results["mt_pe_unchanged_words"] = edit_ops["unchanged_words"]
    results["mt_pe_changed_original_words"] = edit_ops["changed_original_words"]
    results["mt_pe_unchanged_ratio"] = edit_ops["unchanged_ratio"]
    results["mt_pe_changed_ratio_original"] = edit_ops["changed_ratio_original"]

    # MT-PE overlap metrics.
    # These measure similarity/change between PE and raw MT,
    # not true translation quality.
    mt_pe_overlap_scores = reference_based_scores(
        candidate_text=post_edited_text,
        reference_text=raw_mt,
    )

    results["mt_pe_overlap_bleu"] = mt_pe_overlap_scores.get("bleu")
    results["mt_pe_overlap_chrf"] = mt_pe_overlap_scores.get("chrf")
    results["mt_pe_overlap_ter"] = mt_pe_overlap_scores.get("ter")

    # Backward-compatible aliases.
    results["mt_pe_bleu"] = results["mt_pe_overlap_bleu"]
    results["mt_pe_chrf"] = results["mt_pe_overlap_chrf"]
    results["mt_pe_ter"] = results["mt_pe_overlap_ter"]

    if use_bert:
        results["mt_pe_bertscore_f1"] = bert_score(
            post_edited_text,
            raw_mt,
            language=bert_language,
        )
    else:
        results["mt_pe_bertscore_f1"] = None

    # --------------------------------------------------------
    # 3. Reference-based quality metrics
    # --------------------------------------------------------

    if reference_text:
        add_reference_based_quality_scores(
            results=results,
            prefix="raw_mt_quality",
            candidate_text=raw_mt,
            reference_text=reference_text,
            source_text=source_text,
            use_bert=use_bert,
            bert_language=bert_language,
            comet_scorer=comet_scorer,
        )

        add_reference_based_quality_scores(
            results=results,
            prefix="pe_quality",
            candidate_text=post_edited_text,
            reference_text=reference_text,
            source_text=source_text,
            use_bert=use_bert,
            bert_language=bert_language,
            comet_scorer=comet_scorer,
        )

        if human_translation:
            add_reference_based_quality_scores(
                results=results,
                prefix="ht_quality",
                candidate_text=human_translation,
                reference_text=reference_text,
                source_text=source_text,
                use_bert=use_bert,
                bert_language=bert_language,
                comet_scorer=comet_scorer,
            )
        else:
            results["ht_quality_bleu"] = None
            results["ht_quality_chrf"] = None
            results["ht_quality_ter"] = None
            results["ht_quality_bertscore_f1"] = None
            results["ht_quality_comet"] = None

    else:
        for prefix in ["raw_mt_quality", "pe_quality", "ht_quality"]:
            results[f"{prefix}_bleu"] = None
            results[f"{prefix}_chrf"] = None
            results[f"{prefix}_ter"] = None
            results[f"{prefix}_bertscore_f1"] = None
            results[f"{prefix}_comet"] = None

    # --------------------------------------------------------
    # 4. Teacher assessment
    # --------------------------------------------------------

    results["teacher_score"] = teacher_score
    results["teacher_feedback"] = teacher_feedback if teacher_feedback else None

    # --------------------------------------------------------
    # 5. Interpretation
    # --------------------------------------------------------

    interpretation = []

    cosine = results["mt_pe_cosine_similarity"]
    edit_ratio = results["mt_pe_edit_distance_ratio"]
    overlap_ter = results["mt_pe_overlap_ter"]
    changed_ratio = results["mt_pe_changed_ratio_original"]

    if cosine is not None:
        if cosine >= 0.90:
            interpretation.append(
                "Very close to raw MT; minimal semantic change detected."
            )
        elif cosine >= 0.75:
            interpretation.append(
                "Moderate similarity to raw MT; some semantic change detected."
            )
        else:
            interpretation.append(
                "Low similarity to raw MT; substantial semantic change detected."
            )

    if edit_ratio is not None:
        if edit_ratio < 0.10:
            interpretation.append("Low edit-distance ratio.")
        elif edit_ratio < 0.35:
            interpretation.append("Moderate edit-distance ratio.")
        else:
            interpretation.append("High edit-distance ratio.")

    if changed_ratio is not None:
        if changed_ratio < 0.10:
            interpretation.append("Most raw MT wording was retained.")
        elif changed_ratio < 0.35:
            interpretation.append("A moderate proportion of raw MT wording was changed.")
        else:
            interpretation.append("A large proportion of raw MT wording was changed.")

    if overlap_ter is not None:
        if overlap_ter < 20:
            interpretation.append("Low TER against raw MT; limited surface editing.")
        elif overlap_ter < 50:
            interpretation.append("Moderate TER against raw MT.")
        else:
            interpretation.append("High TER against raw MT; substantial rewriting.")

    if reference_text:
        interpretation.append(
            "Reference-based quality metrics were calculated against an independent reference."
        )
    else:
        interpretation.append(
            "No independent reference was provided, so reference-based quality metrics are unavailable."
        )

    if teacher_score is not None:
        interpretation.append(
            "Teacher score is available and should be treated as the main human quality indicator."
        )

    results["mt_pe_interpretation"] = " ".join(interpretation)

    return results


# ============================================================
# Lightweight compatibility function
# ============================================================

def compare_mt_pe(mt_output, post_edited_text):
    """
    Lightweight MT-PE comparison.

    Use this if you only need simple descriptive metrics.
    """

    mt_output = safe_text(mt_output)
    post_edited_text = safe_text(post_edited_text)

    mt_words = word_count(mt_output)
    pe_words = word_count(post_edited_text)
    similarity = lexical_similarity(mt_output, post_edited_text)
    edit_ops = word_edit_operations(mt_output, post_edited_text)

    return {
        "mt_word_count": mt_words,
        "pe_word_count": pe_words,
        "word_count_difference": pe_words - mt_words,
        "lexical_similarity": round(similarity, 3),
        "change_ratio": round(1 - similarity, 3),
        "inserted_words": edit_ops["inserted_words"],
        "deleted_words": edit_ops["deleted_words"],
        "replaced_words": edit_ops["replaced_words"],
        "unchanged_words": edit_ops["unchanged_words"],
        "unchanged_ratio": edit_ops["unchanged_ratio"],
        "changed_ratio_original": edit_ops["changed_ratio_original"],
    }


# ============================================================
# Supabase helper
# ============================================================

def build_research_metrics_payload(results, research_mode=True):
    """
    Converts results dictionary into fields ready to save in Supabase.

    Use this in your app when building the submission dictionary.
    """

    return {
        "research_mode": research_mode,
        "advanced_metrics_status": "pending" if research_mode else "not_required",

        "raw_mt_word_count": results.get("raw_mt_word_count"),
        "ht_word_count": results.get("ht_word_count"),
        "pe_word_count": results.get("pe_word_count"),
        "reference_word_count": results.get("reference_word_count"),
        "mt_pe_word_count_difference": results.get("mt_pe_word_count_difference"),

        "mt_pe_cosine_similarity": results.get("mt_pe_cosine_similarity"),
        "mt_pe_edit_distance_ratio": results.get("mt_pe_edit_distance_ratio"),
        "mt_pe_length_ratio": results.get("mt_pe_length_ratio"),
        "mt_pe_lexical_similarity": results.get("mt_pe_lexical_similarity"),
        "mt_pe_change_ratio": results.get("mt_pe_change_ratio"),

        "mt_pe_inserted_words": results.get("mt_pe_inserted_words"),
        "mt_pe_deleted_words": results.get("mt_pe_deleted_words"),
        "mt_pe_replaced_words": results.get("mt_pe_replaced_words"),
        "mt_pe_replacement_output_words": results.get("mt_pe_replacement_output_words"),
        "mt_pe_unchanged_words": results.get("mt_pe_unchanged_words"),
        "mt_pe_changed_original_words": results.get("mt_pe_changed_original_words"),
        "mt_pe_unchanged_ratio": results.get("mt_pe_unchanged_ratio"),
        "mt_pe_changed_ratio_original": results.get("mt_pe_changed_ratio_original"),

        "mt_pe_overlap_bleu": results.get("mt_pe_overlap_bleu"),
        "mt_pe_overlap_chrf": results.get("mt_pe_overlap_chrf"),
        "mt_pe_overlap_ter": results.get("mt_pe_overlap_ter"),

        "raw_mt_quality_bleu": results.get("raw_mt_quality_bleu"),
        "raw_mt_quality_chrf": results.get("raw_mt_quality_chrf"),
        "raw_mt_quality_ter": results.get("raw_mt_quality_ter"),

        "pe_quality_bleu": results.get("pe_quality_bleu"),
        "pe_quality_chrf": results.get("pe_quality_chrf"),
        "pe_quality_ter": results.get("pe_quality_ter"),

        "ht_quality_bleu": results.get("ht_quality_bleu"),
        "ht_quality_chrf": results.get("ht_quality_chrf"),
        "ht_quality_ter": results.get("ht_quality_ter"),

        "teacher_score": results.get("teacher_score"),
        "teacher_feedback": results.get("teacher_feedback"),

        "raw_mt_quality_bertscore_f1": results.get("raw_mt_quality_bertscore_f1"),
        "pe_quality_bertscore_f1": results.get("pe_quality_bertscore_f1"),
        "ht_quality_bertscore_f1": results.get("ht_quality_bertscore_f1"),

        "raw_mt_quality_comet": results.get("raw_mt_quality_comet"),
        "pe_quality_comet": results.get("pe_quality_comet"),
        "ht_quality_comet": results.get("ht_quality_comet"),

        "mt_pe_interpretation": results.get("mt_pe_interpretation"),
    }
