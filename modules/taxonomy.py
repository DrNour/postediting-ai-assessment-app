ERROR_TAXONOMY = {
    "accuracy": [
        "mistranslation",
        "omission",
        "addition",
        "untranslated_text",
        "wrong_number",
        "wrong_date",
        "wrong_reference",
    ],
    "terminology": [
        "wrong_term",
        "inconsistent_term",
        "overly_literal_term",
    ],
    "fluency": [
        "grammar",
        "word_order",
        "spelling",
        "punctuation",
        "awkward_expression",
        "cohesion",
    ],
    "style_register": [
        "too_formal",
        "too_informal",
        "genre_mismatch",
        "unnatural_style",
    ],
    "locale_cultural": [
        "cultural_mismatch",
        "inappropriate_equivalent",
        "locale_format_error",
    ],
    "formatting": [
        "missing_formatting",
        "extra_formatting",
        "spacing",
        "capitalisation",
    ],
}

SEVERITY_LEVELS = ["minor", "major", "critical"]

RUBRIC = {
    "accuracy": {
        "weight": 0.35,
        "description": "Meaning is preserved accurately.",
    },
    "fluency": {
        "weight": 0.20,
        "description": "The translation is grammatically correct and natural.",
    },
    "terminology": {
        "weight": 0.20,
        "description": "Key terms are translated correctly and consistently.",
    },
    "style_register": {
        "weight": 0.15,
        "description": "The translation matches the appropriate tone and genre.",
    },
    "mechanics_formatting": {
        "weight": 0.10,
        "description": "Spelling, punctuation, layout, and formatting are acceptable.",
    },
}
