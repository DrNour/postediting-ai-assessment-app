import re
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_english(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def semantic_similarity(text1, text2):
    model = get_model()
    embeddings = model.encode([str(text1), str(text2)], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(score[0][0]), 3)


def meaning_shift_warning(score):
    if score is None:
        return "not_available"
    if score >= 0.85:
        return "low"
    if score >= 0.70:
        return "medium"
    return "high"
