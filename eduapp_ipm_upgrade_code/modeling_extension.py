from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grouped-CV feature ablations for EduApp-PE.")
    p.add_argument("--input", required=True, help="Enriched CSV path")
    p.add_argument("--target", default="preferred_edit_count")
    p.add_argument("--group", default="participant_id")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def feature_sets(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    length = existing_cols(df, ["source_words", "mt_words", "source_chars", "mt_chars"])
    task_cat = existing_cols(df, ["exercise_id"])
    telemetry = existing_cols(
        df,
        [
            "time_spent_sec",
            "submission_words",
            "submission_chars",
            "submission_words_per_min",
            "submission_chars_per_min",
            "characters_recorded",
        ],
    )
    revision = existing_cols(
        df,
        [
            "mt_postedit_edit_distance",
            "mt_postedit_edit_ratio_char",
            "source_submission_edit_distance",
            "source_submission_edit_ratio_char",
            "length_ratio_submission_to_mt_words",
        ],
    )
    neural = existing_cols(
        df,
        [
            "mt_vs_postedit_bleu",
            "mt_vs_postedit_chrf",
            "mt_vs_postedit_chrfpp",
            "mt_vs_postedit_ter",
            "mt_vs_postedit_bertscore_precision",
            "mt_vs_postedit_bertscore_recall",
            "mt_vs_postedit_bertscore_f1",
            "comet_ref_score",
            "comet_qe_score",
        ],
    )
    embeddings = existing_cols(
        df,
        [
            "embedding_cosine_source_mt",
            "embedding_cosine_source_submission",
            "embedding_cosine_mt_submission",
        ],
    )
    llm = existing_cols(
        df,
        [
            "llm_adequacy",
            "llm_fluency",
            "llm_terminology",
            "llm_style",
            "llm_overall_quality",
            "llm_estimated_effort_numeric",
            "llm_severity_numeric",
            "llm_error_span_count",
            "llm_major_or_critical_error_count",
        ],
    )

    return {
        "F0_dummy": {"numeric": [], "categorical": []},
        "F1_pre_task_length": {"numeric": length, "categorical": []},
        "F2_pre_task_length_exercise": {"numeric": length, "categorical": task_cat},
        "F3_telemetry_output": {"numeric": length + telemetry, "categorical": task_cat},
        "F4_revision_audit": {"numeric": length + telemetry + revision, "categorical": task_cat},
        "F5_neural_qe_metrics": {"numeric": length + telemetry + revision + neural, "categorical": task_cat},
        "F6_embedding_metrics": {"numeric": length + telemetry + revision + neural + embeddings, "categorical": task_cat},
        "F7_llm_judge_metrics": {"numeric": length + telemetry + revision + neural + embeddings + llm, "categorical": task_cat},
        "F8_all_sota_features": {"numeric": length + telemetry + revision + neural + embeddings + llm, "categorical": task_cat},
    }


def preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def make_regression_models(has_features: bool) -> dict[str, Any]:
    if not has_features:
        return {"dummy_median": DummyRegressor(strategy="median")}
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }


def make_classification_models(has_features: bool) -> dict[str, Any]:
    if not has_features:
        return {"dummy_majority": DummyClassifier(strategy="most_frequent")}
    return {
        "logistic_balanced": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }


def summarize(values: list[float]) -> str:
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "NA"
    return f"{np.mean(values):.3f} +/- {np.std(values):.3f}"


def evaluate_regression(df: pd.DataFrame, target: str, group_col: str, folds: int) -> pd.DataFrame:
    sets = feature_sets(df)
    y = df[target].astype(float).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    rows = []

    for fs_name, spec in sets.items():
        numeric = spec["numeric"]
        categorical = spec["categorical"]
        features = numeric + categorical
        has_features = bool(features)
        X = df[features] if has_features else pd.DataFrame(index=df.index)
        for model_name, model in make_regression_models(has_features).items():
            maes, rmses, r2s = [], [], []
            for train_idx, test_idx in cv.split(X, y, groups):
                if has_features:
                    pipe = Pipeline([("prep", preprocessor(numeric, categorical)), ("model", model)])
                    pipe.fit(X.iloc[train_idx], y[train_idx])
                    pred = pipe.predict(X.iloc[test_idx])
                else:
                    model.fit(np.zeros((len(train_idx), 1)), y[train_idx])
                    pred = model.predict(np.zeros((len(test_idx), 1)))
                maes.append(mean_absolute_error(y[test_idx], pred))
                rmses.append(mean_squared_error(y[test_idx], pred, squared=False))
                r2s.append(r2_score(y[test_idx], pred))
            rows.append(
                {
                    "feature_set": fs_name,
                    "model": model_name,
                    "n_features": len(features),
                    "MAE": summarize(maes),
                    "RMSE": summarize(rmses),
                    "R2": summarize(r2s),
                }
            )
    return pd.DataFrame(rows)


def evaluate_classification(df: pd.DataFrame, target: str, group_col: str, folds: int) -> pd.DataFrame:
    sets = feature_sets(df)
    y_raw = df[target].astype(float)
    threshold = float(y_raw.median())
    y = (y_raw > threshold).astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    rows = []

    for fs_name, spec in sets.items():
        numeric = spec["numeric"]
        categorical = spec["categorical"]
        features = numeric + categorical
        has_features = bool(features)
        X = df[features] if has_features else pd.DataFrame(index=df.index)
        for model_name, model in make_classification_models(has_features).items():
            accs, baccs, f1s, aucs = [], [], [], []
            for train_idx, test_idx in cv.split(X, y, groups):
                if has_features:
                    pipe = Pipeline([("prep", preprocessor(numeric, categorical)), ("model", model)])
                    pipe.fit(X.iloc[train_idx], y[train_idx])
                    pred = pipe.predict(X.iloc[test_idx])
                    if hasattr(pipe[-1], "predict_proba"):
                        prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
                    else:
                        prob = pred
                else:
                    model.fit(np.zeros((len(train_idx), 1)), y[train_idx])
                    pred = model.predict(np.zeros((len(test_idx), 1)))
                    prob = pred
                accs.append(accuracy_score(y[test_idx], pred))
                baccs.append(balanced_accuracy_score(y[test_idx], pred))
                f1s.append(f1_score(y[test_idx], pred, zero_division=0))
                try:
                    aucs.append(roc_auc_score(y[test_idx], prob))
                except ValueError:
                    aucs.append(float("nan"))
            rows.append(
                {
                    "feature_set": fs_name,
                    "model": model_name,
                    "n_features": len(features),
                    "threshold_edit_count": threshold,
                    "Accuracy": summarize(accs),
                    "Balanced_accuracy": summarize(baccs),
                    "F1": summarize(f1s),
                    "AUC": summarize(aucs),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df[pd.notna(df[args.target]) & pd.notna(df[args.group])].copy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = evaluate_regression(df, args.target, args.group, args.folds)
    clf = evaluate_classification(df, args.target, args.group, args.folds)

    reg.to_csv(out_dir / "grouped_cv_regression_results.csv", index=False)
    clf.to_csv(out_dir / "grouped_cv_classification_results.csv", index=False)

    print(f"Wrote {out_dir / 'grouped_cv_regression_results.csv'}")
    print(f"Wrote {out_dir / 'grouped_cv_classification_results.csv'}")


if __name__ == "__main__":
    main()
