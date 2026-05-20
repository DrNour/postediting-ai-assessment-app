from rapidfuzz import fuzz
def compare_postedit_with_raw_mt(
    raw_mt,
    post_edited_text,
    use_bert=False,
    bert_language="en",
):
    """
    Compares the student's post-edited text with the original raw MT output.

    This is useful for measuring post-editing effort:
    - High cosine similarity = student changed little
    - Low cosine similarity = student changed more
    """

    raw_mt = safe_text(raw_mt)
    post_edited_text = safe_text(post_edited_text)

    results = {}

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

    mt_pe_scores = reference_based_scores(
        post_edited_text,
        raw_mt,
    )

    results["mt_pe_bleu"] = mt_pe_scores.get("bleu")
    results["mt_pe_chrf"] = mt_pe_scores.get("chrf")
    results["mt_pe_ter"] = mt_pe_scores.get("ter")

    if use_bert:
        results["mt_pe_bertscore_f1"] = bert_score(
            post_edited_text,
            raw_mt,
            language=bert_language,
        )
    else:
        results["mt_pe_bertscore_f1"] = None

    interpretation = []

    cosine = results["mt_pe_cosine_similarity"]
    edit_ratio = results["mt_pe_edit_distance_ratio"]
    ter = results["mt_pe_ter"]

    if cosine is not None:
        if cosine >= 0.90:
            interpretation.append("Very close to raw MT; minimal editing detected.")
        elif cosine >= 0.75:
            interpretation.append("Moderate similarity to raw MT; some editing detected.")
        else:
            interpretation.append("Substantial changes from raw MT detected.")

    if edit_ratio is not None:
        if edit_ratio < 0.10:
            interpretation.append("Low edit-distance ratio.")
        elif edit_ratio < 0.35:
            interpretation.append("Moderate edit-distance ratio.")
        else:
            interpretation.append("High edit-distance ratio.")

    if ter is not None:
        if ter < 20:
            interpretation.append("Low TER against raw MT; limited edits.")
        elif ter < 50:
            interpretation.append("Moderate TER against raw MT.")
        else:
            interpretation.append("High TER against raw MT; substantial rewriting.")

    results["mt_pe_interpretation"] = " ".join(interpretation)

    return results

def word_count(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())


def word_count_difference(mt_output, post_edited_text):
    return word_count(post_edited_text) - word_count(mt_output)


def lexical_similarity(mt_output, post_edited_text):
    if not mt_output and not post_edited_text:
        return 1.0
    return fuzz.ratio(str(mt_output), str(post_edited_text)) / 100


def change_ratio(mt_output, post_edited_text):
    return 1 - lexical_similarity(mt_output, post_edited_text)


def compare_mt_pe(mt_output, post_edited_text):
    mt_words = word_count(mt_output)
    pe_words = word_count(post_edited_text)
    similarity = lexical_similarity(mt_output, post_edited_text)
    return {
        "mt_word_count": mt_words,
        "pe_word_count": pe_words,
        "word_count_difference": pe_words - mt_words,
        "lexical_similarity": round(similarity, 3),
        "change_ratio": round(1 - similarity, 3),
    }
