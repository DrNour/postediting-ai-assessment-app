from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .schemas import MetricConfig, MetricResult, SegmentRecord


_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def normalize_text(text: Any) -> str:
    """Normalize text lightly without destroying Arabic/English content."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text)
    text = text.replace("\u200f", "").replace("\u200e", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    text = normalize_text(text)
    return len(_WORD_RE.findall(text))


def char_count(text: str) -> int:
    return len(normalize_text(text))


def levenshtein_distance(a: str, b: str) -> int:
    """Small dependency-free Levenshtein distance implementation."""
    a, b = normalize_text(a), normalize_text(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (ca != cb)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    try:
        if den == 0:
            return None
        return float(num) / float(den)
    except Exception:
        return None


def compute_core_effort_metrics(record: SegmentRecord) -> dict[str, Any]:
    src = normalize_text(record.source_text)
    mt = normalize_text(record.mt_output)
    sub = normalize_text(record.user_submission)

    mt_postedit_edit_distance = levenshtein_distance(mt, sub) if mt or sub else None
    source_submission_edit_distance = levenshtein_distance(src, sub) if src or sub else None

    metrics: dict[str, Any] = {
        "source_chars": char_count(src),
        "mt_chars": char_count(mt),
        "submission_chars": char_count(sub),
        "source_words": word_count(src),
        "mt_words": word_count(mt),
        "submission_words": word_count(sub),
        "mt_postedit_edit_distance": mt_postedit_edit_distance,
        "mt_postedit_edit_ratio_char": safe_div(mt_postedit_edit_distance, max(1, char_count(mt))),
        "source_submission_edit_distance": source_submission_edit_distance,
        "source_submission_edit_ratio_char": safe_div(source_submission_edit_distance, max(1, char_count(src))),
        "length_ratio_submission_to_mt_words": safe_div(word_count(sub), max(1, word_count(mt))),
        "length_ratio_submission_to_source_words": safe_div(word_count(sub), max(1, word_count(src))),
        "time_spent_sec": record.time_spent_sec,
        "preferred_edit_count": record.preferred_edit_count,
    }

    if record.time_spent_sec is not None and record.time_spent_sec > 0:
        metrics.update(
            {
                "submission_words_per_min": word_count(sub) / (record.time_spent_sec / 60),
                "submission_chars_per_min": char_count(sub) / (record.time_spent_sec / 60),
                "edits_per_min": safe_div(record.preferred_edit_count, record.time_spent_sec / 60),
                "seconds_per_edit": safe_div(record.time_spent_sec, record.preferred_edit_count)
                if record.preferred_edit_count not in (None, 0)
                else None,
            }
        )
    return metrics


def compute_sacrebleu_metrics(record: SegmentRecord) -> tuple[dict[str, Any], list[str]]:
    """Compute MT-vs-human-postedit metrics using the post-edit as a reference.

    This is appropriate for analysis, but the manuscript should state clearly that the
    user submission is treated as a post-edited reference, not as an independently
    created gold standard.
    """
    warnings: list[str] = []
    try:
        from sacrebleu.metrics import BLEU, CHRF, TER
    except Exception as exc:  # pragma: no cover
        return {}, [f"sacrebleu unavailable: {exc}"]

    mt = normalize_text(record.mt_output)
    ref = normalize_text(record.user_submission)
    if not mt or not ref:
        return {}, ["sacrebleu skipped: missing MT output or user submission"]

    try:
        bleu = BLEU(effective_order=True).sentence_score(mt, [ref]).score
        chrf = CHRF(word_order=0).sentence_score(mt, [ref]).score
        chrfpp = CHRF(word_order=2).sentence_score(mt, [ref]).score
        ter = TER().sentence_score(mt, [ref]).score
        return {
            "mt_vs_postedit_bleu": bleu,
            "mt_vs_postedit_chrf": chrf,
            "mt_vs_postedit_chrfpp": chrfpp,
            "mt_vs_postedit_ter": ter,
        }, warnings
    except Exception as exc:
        return {}, [f"sacrebleu failed: {exc}"]


def compute_bertscore_metrics(records: list[SegmentRecord], config: MetricConfig) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        from bert_score import score as bert_score
    except Exception as exc:  # pragma: no cover
        return [{} for _ in records], [f"bert-score unavailable: {exc}"]

    candidates = [normalize_text(r.mt_output) for r in records]
    references = [normalize_text(r.user_submission) for r in records]
    valid = [bool(c and ref) for c, ref in zip(candidates, references)]
    if not any(valid):
        return [{} for _ in records], ["BERTScore skipped: no valid MT/reference pairs"]

    metrics = [{} for _ in records]
    valid_candidates = [c for c, ok in zip(candidates, valid) if ok]
    valid_references = [r for r, ok in zip(references, valid) if ok]

    try:
        P, R, F = bert_score(
            valid_candidates,
            valid_references,
            model_type=config.bertscore_model,
            lang="en",
            verbose=False,
            device=config.device,
        )
        j = 0
        for i, ok in enumerate(valid):
            if ok:
                metrics[i] = {
                    "mt_vs_postedit_bertscore_precision": float(P[j]),
                    "mt_vs_postedit_bertscore_recall": float(R[j]),
                    "mt_vs_postedit_bertscore_f1": float(F[j]),
                }
                j += 1
        return metrics, warnings
    except Exception as exc:
        return [{} for _ in records], [f"BERTScore failed: {exc}"]


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def compute_embedding_metrics(records: list[SegmentRecord], config: MetricConfig) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        model = _load_sentence_transformer(config.embedding_model)
    except Exception as exc:  # pragma: no cover
        return [{} for _ in records], [f"sentence-transformers unavailable: {exc}"]

    texts: list[str] = []
    for r in records:
        texts.extend([normalize_text(r.source_text), normalize_text(r.mt_output), normalize_text(r.user_submission)])

    try:
        emb = model.encode(texts, batch_size=config.batch_size, normalize_embeddings=False, show_progress_bar=False)
    except Exception as exc:
        return [{} for _ in records], [f"embedding computation failed: {exc}"]

    results: list[dict[str, Any]] = []
    for i in range(len(records)):
        src_e, mt_e, sub_e = emb[3 * i], emb[3 * i + 1], emb[3 * i + 2]
        results.append(
            {
                "embedding_cosine_source_mt": _cosine(src_e, mt_e),
                "embedding_cosine_source_submission": _cosine(src_e, sub_e),
                "embedding_cosine_mt_submission": _cosine(mt_e, sub_e),
            }
        )
    return results, warnings


@lru_cache(maxsize=4)
def _load_comet_model(model_name: str, cache_dir: str | None = None):
    from comet import download_model, load_from_checkpoint

    path = download_model(model_name, saving_directory=cache_dir)
    return load_from_checkpoint(path)


def compute_comet_metrics(records: list[SegmentRecord], config: MetricConfig, qe: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    """Compute COMET or COMETKiwi scores.

    qe=False: reference-based COMET uses source, MT, and user_submission as reference.
    qe=True: reference-free QE uses source and MT only.
    """
    warnings: list[str] = []
    model_name = config.qe_model if qe else config.comet_model
    try:
        model = _load_comet_model(model_name, config.cache_dir)
    except Exception as exc:  # pragma: no cover
        return [{} for _ in records], [f"COMET model unavailable ({model_name}): {exc}"]

    data = []
    valid_idx = []
    for i, r in enumerate(records):
        src = normalize_text(r.source_text)
        mt = normalize_text(r.mt_output)
        ref = normalize_text(r.user_submission)
        if qe and src and mt:
            data.append({"src": src, "mt": mt})
            valid_idx.append(i)
        elif not qe and src and mt and ref:
            data.append({"src": src, "mt": mt, "ref": ref})
            valid_idx.append(i)

    out = [{} for _ in records]
    if not data:
        return out, [f"{model_name} skipped: no valid rows"]

    try:
        gpus = 1 if config.device and "cuda" in config.device.lower() else 0
        pred = model.predict(data, batch_size=config.batch_size, gpus=gpus)
        scores = pred.scores
        key = "comet_qe_score" if qe else "comet_ref_score"
        for i, score in zip(valid_idx, scores):
            out[i] = {key: float(score)}
        return out, warnings
    except Exception as exc:
        return [{} for _ in records], [f"COMET prediction failed ({model_name}): {exc}"]


def compute_metrics_for_record(record: SegmentRecord, config: MetricConfig | None = None) -> MetricResult:
    config = config or MetricConfig()
    metrics = compute_core_effort_metrics(record)
    warnings: list[str] = []

    if config.use_sacrebleu:
        extra, warn = compute_sacrebleu_metrics(record)
        metrics.update(extra)
        warnings.extend(warn)

    # Heavy batch metrics should be run with compute_metrics_for_dataframe.
    if config.use_bertscore or config.use_comet or config.use_qe or config.use_embeddings:
        warnings.append("Heavy model metrics are batch-oriented; use compute_metrics_for_dataframe for efficiency.")

    return MetricResult(instance_id=record.instance_id, metrics=metrics, warnings=warnings)


def row_to_record(row: pd.Series) -> SegmentRecord:
    return SegmentRecord(
        instance_id=str(row.get("instance_id", "")) if pd.notna(row.get("instance_id", "")) else None,
        participant_id=str(row.get("participant_id", "")) if pd.notna(row.get("participant_id", "")) else None,
        exercise_id=str(row.get("exercise_id", "")) if pd.notna(row.get("exercise_id", "")) else None,
        task_type=str(row.get("task_type", "unknown")) if pd.notna(row.get("task_type", "unknown")) else "unknown",
        source_text=normalize_text(row.get("source_text", "")),
        mt_output=normalize_text(row.get("mt_output", "")),
        user_submission=normalize_text(row.get("user_submission", "")),
        time_spent_sec=float(row["time_spent_sec"]) if "time_spent_sec" in row and pd.notna(row["time_spent_sec"]) else None,
        preferred_edit_count=float(row["preferred_edit_count"])
        if "preferred_edit_count" in row and pd.notna(row["preferred_edit_count"])
        else None,
    )


def compute_metrics_for_dataframe(df: pd.DataFrame, config: MetricConfig | None = None) -> pd.DataFrame:
    config = config or MetricConfig()
    records = [row_to_record(row) for _, row in df.iterrows()]

    all_metrics: list[dict[str, Any]] = []
    warnings: list[str] = []
    for r in records:
        res = compute_metrics_for_record(r, config)
        all_metrics.append(res.metrics)
        warnings.extend(res.warnings)

    if config.use_bertscore:
        bert_metrics, warn = compute_bertscore_metrics(records, config)
        warnings.extend(warn)
        for m, extra in zip(all_metrics, bert_metrics):
            m.update(extra)

    if config.use_embeddings:
        emb_metrics, warn = compute_embedding_metrics(records, config)
        warnings.extend(warn)
        for m, extra in zip(all_metrics, emb_metrics):
            m.update(extra)

    if config.use_comet:
        comet_metrics, warn = compute_comet_metrics(records, config, qe=False)
        warnings.extend(warn)
        for m, extra in zip(all_metrics, comet_metrics):
            m.update(extra)

    if config.use_qe:
        qe_metrics, warn = compute_comet_metrics(records, config, qe=True)
        warnings.extend(warn)
        for m, extra in zip(all_metrics, qe_metrics):
            m.update(extra)

    metrics_df = pd.DataFrame(all_metrics)
    out = pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
    if warnings:
        out.attrs["warnings"] = sorted(set(warnings))
    return out
