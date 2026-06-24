from rapidfuzz import fuzz
from difflib import SequenceMatcher
import re


def word_count(text):
    """
    Simple whitespace-based word count.
    Works reasonably for English and Arabic, but can be replaced later
    with a stronger tokenizer if needed.
    """
    text = safe_text(text)
    if not text:
        return 0
    return len(text.split())


def word_count_difference(original_text, revised_text):
    return word_count(revised_text) - word_count(original_text)


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
    """
    return 1 - lexical_similarity(text_a, text_b)


def tokenize_words(text):
    """
    Unicode-aware word tokenizer.

    This is still simple, but better than plain split() for counting
    word-level insertions, deletions, replacements, and unchanged words.
    """
    text = safe_text(text)
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def word_edit_operations(original_text, revised_text):
    """
    Counts inserted, deleted, replaced, and unchanged words from
    original_text to revised_text.

    In this project:
    - original_text is usually raw MT
    - revised_text is usually post-edited MT

    replaced_words counts original-side words that were replaced.
    replacement_output_words counts revised-side words that replaced them.
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

    candidate_text = MT, HT, or PE output
    reference_text = independent human/reference translation

    Important:
    These are quality-oriented only when reference_text is an independent
    reference, not the raw MT.
    """

    candidate_text = safe_text(candidate_text)
    reference_text = safe_text(reference_text)

    if not candidate_text or not reference_text:
        results[f"{prefix}_bleu"] = None
        results[f"{prefix}_chrf"] = None
        results[f"{prefix}_ter"] = None
        results[f"{prefix}_bertscore_f1"] = None
        results[f"{prefix}_comet"] = None
        return results

    scores = reference_based_scores(
        candidate_text,
        reference_text,
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
            results[f"{prefix}_comet"] = comet_scorer(
                source_text,
                candidate_text,
                reference_text,
            )
    else:
        results[f"{prefix}_comet"] = None

    return results


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

    This function separates three layers:

    1. MT-to-PE effort:
       How much did the student change the raw MT?

    2. Reference-based quality:
       How good are MT, HT, and PE against an independent reference?

    3. Teacher assessment:
       Human score and feedback.

    Parameters
    ----------
    raw_mt : str
        Original machine translation output.

    post_edited_text : str
        Student's post-edited version of the raw MT.

    human_translation : str, optional
        Student's human translation, if available.

    reference_text : str, optional
        Independent reference translation.
        Needed for true quality metrics.

    source_text : str, optional
        Source text. Needed for COMET if using a COMET scorer.

    teacher_score : float or int, optional
        Teacher's score.

    teacher_feedback : str, optional
        Teacher's qualitative feedback.

    use_bert : bool
        Whether to calculate BERTScore.

    bert_language : str
        Language code for BERTScore. For Arabic, use "ar".
        For English, use "en".

    comet_scorer : callable, optional
        Optional COMET scoring function.
        Expected signature:
        comet_scorer(source, candidate, reference)

    Returns
    -------
    dict
        Dictionary of effort, quality, and teacher-assessment metrics.
    """

    raw_mt = safe_text(raw_mt)
    post_edited_text = safe_text(post_edited_text)
    human_translation = safe_text(human_translation)
    reference_text = safe_text(reference_text)
    source_text = safe_text(source_text)
    teacher_feedback = safe_text(teacher_feedback)

    results = {}

    # ---------------------------------------------------------
    # 1. Basic word counts
    # ---------------------------------------------------------

    results["raw_mt_word_count"] = word_count(raw_mt)
    results["pe_word_count"] = word_count(post_edited_text)
    results["mt_pe_word_count_difference"] = word_count_difference(
        raw_mt,
        post_edited_text,
    )

    if human_translation:
        results["ht_word_count"] = word_count(human_translation)
    else:
        results["ht_word_count"] = None

    if reference_text:
        results["reference_word_count"] = word_count(reference_text)
    else:
        results["reference_word_count"] = None

    # ---------------------------------------------------------
    # 2. MT-to-PE post-editing effort metrics
    # ---------------------------------------------------------

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

    # These are not true quality metrics because raw MT is being used
    # as the comparison text. They measure overlap/change from MT to PE.
    mt_pe_overlap_scores = reference_based_scores(
        post_edited_text,
        raw_mt,
    )

    results["mt_pe_overlap_bleu"] = mt_pe_overlap_scores.get("bleu")
    results["mt_pe_overlap_chrf"] = mt_pe_overlap_scores.get("chrf")
    results["mt_pe_overlap_ter"] = mt_pe_overlap_scores.get("ter")

    # Backward-compatible aliases.
    # Keep these if older parts of your app expect these column names.
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

    # ---------------------------------------------------------
    # 3. Reference-based quality metrics
    # ---------------------------------------------------------
    # These are quality-oriented only if reference_text is independent.

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
        results["raw_mt_quality_bleu"] = None
        results["raw_mt_quality_chrf"] = None
        results["raw_mt_quality_ter"] = None
        results["raw_mt_quality_bertscore_f1"] = None
        results["raw_mt_quality_comet"] = None

        results["pe_quality_bleu"] = None
        results["pe_quality_chrf"] = None
        results["pe_quality_ter"] = None
        results["pe_quality_bertscore_f1"] = None
        results["pe_quality_comet"] = None

        results["ht_quality_bleu"] = None
        results["ht_quality_chrf"] = None
        results["ht_quality_ter"] = None
        results["ht_quality_bertscore_f1"] = None
        results["ht_quality_comet"] = None

    # ---------------------------------------------------------
    # 4. Teacher score and feedback
    # ---------------------------------------------------------

    results["teacher_score"] = teacher_score
    results["teacher_feedback"] = teacher_feedback if teacher_feedback else None

    # ---------------------------------------------------------
    # 5. Interpretation of MT-to-PE effort
    # ---------------------------------------------------------

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
            "No independent reference was provided, so BLEU, chrF, TER, BERTScore, and COMET quality scores are unavailable."
        )

    if teacher_score is not None:
        interpretation.append(
            "Teacher score is available and should be treated as the main human quality indicator."
        )

    results["mt_pe_interpretation"] = " ".join(interpretation)

    return results


def compare_mt_pe(mt_output, post_edited_text):
    """
    Lightweight compatibility function.

    Use this when you only need simple MT-PE descriptive metrics.
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
