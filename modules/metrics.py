from rapidfuzz import fuzz


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
