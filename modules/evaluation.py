import json

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score


def extract_first_ai_category(ai_possible_errors):
    if pd.isna(ai_possible_errors):
        return None
    try:
        errors = json.loads(ai_possible_errors)
        if isinstance(errors, list) and len(errors) > 0:
            return errors[0].get("category")
    except Exception:
        return None
    return None


def extract_first_ai_confidence(ai_possible_errors):
    if pd.isna(ai_possible_errors):
        return None
    try:
        errors = json.loads(ai_possible_errors)
        if isinstance(errors, list) and len(errors) > 0:
            return errors[0].get("confidence")
    except Exception:
        return None
    return None


def prepare_evaluation_df(df):
    eval_df = df.copy()
    if "ai_possible_errors" in eval_df.columns:
        eval_df["ai_category"] = eval_df["ai_possible_errors"].apply(extract_first_ai_category)
        eval_df["ai_confidence"] = eval_df["ai_possible_errors"].apply(extract_first_ai_confidence)
    else:
        eval_df["ai_category"] = None
        eval_df["ai_confidence"] = None
    return eval_df


def calculate_review_status_counts(df):
    if "review_status" not in df.columns:
        return pd.DataFrame()
    review_df = df.dropna(subset=["review_status"]).copy()
    if review_df.empty:
        return pd.DataFrame()
    counts = review_df["review_status"].value_counts().reset_index()
    counts.columns = ["review_status", "count"]
    total = counts["count"].sum()
    counts["percentage"] = round(counts["count"] / total * 100, 2)
    return counts


def calculate_ai_teacher_agreement(df):
    eval_df = prepare_evaluation_df(df)
    comparison_df = eval_df.dropna(subset=["teacher_category", "ai_category"]).copy()

    if comparison_df.empty:
        return {"error": "No comparable teacher and AI labels available."}

    teacher_labels = comparison_df["teacher_category"]
    ai_labels = comparison_df["ai_category"]
    kappa = cohen_kappa_score(teacher_labels, ai_labels)
    report = classification_report(teacher_labels, ai_labels, output_dict=True, zero_division=0)

    return {
        "kappa": kappa,
        "classification_report": report,
        "comparison_df": comparison_df,
    }


def summarise_usefulness(df):
    if "usefulness_rating" not in df.columns:
        return {"error": "No usefulness rating column available."}
    ratings = pd.to_numeric(df["usefulness_rating"], errors="coerce").dropna()
    if ratings.empty:
        return {"error": "No usefulness ratings available."}
    return {
        "count": int(ratings.count()),
        "mean": round(float(ratings.mean()), 2),
        "std": round(float(ratings.std()), 2),
        "median": round(float(ratings.median()), 2),
        "min": int(ratings.min()),
        "max": int(ratings.max()),
    }


def usefulness_by_review_status(df):
    if "review_status" not in df.columns or "usefulness_rating" not in df.columns:
        return pd.DataFrame()
    useful_df = df.dropna(subset=["review_status", "usefulness_rating"]).copy()
    if useful_df.empty:
        return pd.DataFrame()
    useful_df["usefulness_rating"] = pd.to_numeric(useful_df["usefulness_rating"], errors="coerce")
    summary = useful_df.groupby("review_status")["usefulness_rating"].agg(
        ["count", "mean", "std", "median"]
    ).reset_index()
    summary["mean"] = summary["mean"].round(2)
    summary["std"] = summary["std"].round(2)
    summary["median"] = summary["median"].round(2)
    return summary
