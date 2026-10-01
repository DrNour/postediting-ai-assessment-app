"""Teacher-only, source-aware analytics for EduApp-PE."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from supabase import create_client

from modules.auth import require_teacher_access
from modules.task_mode import POST_EDITING, normalize_task_type, task_type_label

try:
    import sacrebleu
except Exception:
    sacrebleu = None

ANALYTICS_VERSION = "EduApp-PE advanced analytics v1.0"
TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:[.,]\d+)?|[^\w\s]", re.UNICODE)

st.title("Advanced Research Analytics")
st.write("Source-aware, task-aware analytics for Arabic-English translation and post-editing research.")
st.info("MT-to-post-edit comparisons measure editing effort. Translation quality metrics are calculated only when an independent reference translation is available.")

if not require_teacher_access("advanced_research_analytics"):
    st.stop()

@st.cache_resource
def get_supabase_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_data(ttl=60)
def load_submissions():
    response = get_supabase_client().table("submissions").select("*").range(0, 9999).execute()
    return pd.DataFrame(response.data or [])

def first_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None

def clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()

def tokens(text):
    return TOKEN_RE.findall(clean_text(text))

def edit_operations(mt_text, output_text):
    mt_tokens, output_tokens = tokens(mt_text), tokens(output_text)
    matcher = SequenceMatcher(a=mt_tokens, b=output_tokens)
    inserted = deleted = replaced = unchanged = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
        elif tag == "insert":
            inserted += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "replace":
            replaced += max(i2 - i1, j2 - j1)
    distance = inserted + deleted + replaced
    return {
        "mt_token_count": len(mt_tokens),
        "output_token_count": len(output_tokens),
        "insertions": inserted,
        "deletions": deleted,
        "replacements": replaced,
        "unchanged_tokens": unchanged,
        "edit_operations": distance,
        "edit_ratio": round(distance / max(len(mt_tokens), 1), 4),
        "unchanged_ratio": round(unchanged / max(len(mt_tokens), 1), 4),
        "length_ratio": round(len(output_tokens) / max(len(mt_tokens), 1), 4),
    }

def reference_metrics(reference, output):
    if not clean_text(reference) or not clean_text(output) or sacrebleu is None:
        return {"reference_chrf": None, "reference_ter": None}
    return {
        "reference_chrf": round(sacrebleu.sentence_chrf(output, [reference]).score, 3),
        "reference_ter": round(sacrebleu.sentence_ter(output, [reference]).score, 3),
    }

def validation_flags(source, mt, output, reference, task_type):
    flags = []
    if not source:
        flags.append("missing_source")
    if not output:
        flags.append("missing_student_output")
    if task_type == POST_EDITING and not mt:
        flags.append("missing_mt_for_post_editing")
    if task_type != POST_EDITING and mt:
        flags.append("mt_present_in_translation_task")
    if task_type == POST_EDITING and mt and output and mt == output:
        flags.append("unchanged_mt_output")
    if reference and not output:
        flags.append("reference_without_student_output")
    return "; ".join(flags) if flags else "valid"

raw_df = load_submissions()
if raw_df.empty:
    st.warning("No submissions found.")
    st.stop()

source_col = first_column(raw_df, ["source_text", "source", "source_segment"])
mt_col = first_column(raw_df, ["machine_translation", "raw_mt", "mt_output"])
output_col = first_column(raw_df, ["post_edited_text", "student_submission", "final_submission"])
reference_col = first_column(raw_df, ["reference_translation", "reference_text", "reference"])
time_col = first_column(raw_df, ["time_spent_sec", "editing_time_seconds", "editing_time_sec"])
condition_col = first_column(raw_df, ["condition", "mt_condition", "workflow_condition", "assignment_code"])

st.caption(
    "Detected fields: "
    + ", ".join(
        f"{label} = {column or 'not found'}"
        for label, column in [
            ("source", source_col), ("MT", mt_col), ("student output", output_col),
            ("reference", reference_col), ("time", time_col), ("condition", condition_col),
        ]
    )
)

if not source_col or not output_col:
    st.error("This dataset needs source_text and a student-output field before source-aware analysis can run.")
    st.stop()

work_df = raw_df.copy()
if "task_type" not in work_df.columns:
    work_df["task_type"] = POST_EDITING
work_df["task_type"] = work_df["task_type"].apply(normalize_task_type)
work_df["task_label"] = work_df["task_type"].apply(task_type_label)

with st.sidebar:
    st.header("Research filters")
    task_options = sorted(work_df["task_label"].dropna().unique().tolist())
    selected_tasks = st.multiselect("Task type", task_options, default=task_options)
    if selected_tasks:
        work_df = work_df[work_df["task_label"].isin(selected_tasks)]
    if condition_col:
        conditions = sorted(work_df[condition_col].dropna().astype(str).unique().tolist())
        selected_conditions = st.multiselect("Condition / workflow", conditions, default=conditions)
        if selected_conditions:
            work_df = work_df[work_df[condition_col].astype(str).isin(selected_conditions)]

rows = []
for _, row in work_df.iterrows():
    source = clean_text(row.get(source_col))
    mt = clean_text(row.get(mt_col)) if mt_col else ""
    output = clean_text(row.get(output_col))
    reference = clean_text(row.get(reference_col)) if reference_col else ""
    task_type = normalize_task_type(row.get("task_type"))
    record = row.to_dict()
    record.update(
        {
            "analytics_source_present": bool(source),
            "analytics_mt_present": bool(mt),
            "analytics_output_present": bool(output),
            "analytics_reference_present": bool(reference),
            "analytics_validation": validation_flags(source, mt, output, reference, task_type),
            "analytics_version": ANALYTICS_VERSION,
        }
    )
    if task_type == POST_EDITING and mt and output:
        record.update(edit_operations(mt, output))
    else:
        record.update(
            {"mt_token_count": None, "output_token_count": len(tokens(output)), "insertions": None,
             "deletions": None, "replacements": None, "unchanged_tokens": None,
             "edit_operations": None, "edit_ratio": None, "unchanged_ratio": None, "length_ratio": None}
        )
    if reference and output:
        record.update(reference_metrics(reference, output))
    else:
        record.update({"reference_chrf": None, "reference_ter": None})
    if time_col:
        time_value = pd.to_numeric(pd.Series([row.get(time_col)]), errors="coerce").iloc[0]
        record["time_spent_sec"] = time_value
        record["edits_per_minute"] = (
            round(record["edit_operations"] / (time_value / 60), 3)
            if pd.notna(time_value) and time_value > 0 and pd.notna(record["edit_operations"])
            else None
        )
    rows.append(record)

analysis_df = pd.DataFrame(rows)
valid_count = int((analysis_df["analytics_validation"] == "valid").sum())
post_edit_mask = analysis_df["task_type"] == POST_EDITING
reference_mask = analysis_df["analytics_reference_present"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected records", len(analysis_df))
m2.metric("Valid records", valid_count)
m3.metric("Post-editing records", int(post_edit_mask.sum()))
m4.metric("Reference-eligible records", int(reference_mask.sum()))

st.subheader("Data quality and representation checks")
validation_table = (
    analysis_df["analytics_validation"].value_counts(dropna=False).rename_axis("validation_status").reset_index(name="records")
)
st.dataframe(validation_table, use_container_width=True, hide_index=True)
st.caption("Records with track-changes or extraction artefacts should be checked before edit-distance or timing results are interpreted.")

st.subheader("Effort and quality separation")
effort_columns = ["edit_ratio", "edit_operations", "insertions", "deletions", "replacements", "unchanged_ratio", "time_spent_sec", "edits_per_minute"]
quality_columns = ["reference_chrf", "reference_ter"]
available_effort = [column for column in effort_columns if column in analysis_df.columns and analysis_df[column].notna().any()]
available_quality = [column for column in quality_columns if column in analysis_df.columns and analysis_df[column].notna().any()]

left, right = st.columns(2)
with left:
    st.markdown("**Post-editing effort**")
    if available_effort:
        st.dataframe(analysis_df.loc[post_edit_mask, available_effort].describe().T.round(3), use_container_width=True)
    else:
        st.info("No usable MT-to-post-edit pairs are available in the selected data.")
with right:
    st.markdown("**Reference-based quality**")
    if available_quality:
        st.dataframe(analysis_df.loc[reference_mask, available_quality].describe().T.round(3), use_container_width=True)
    else:
        st.info("No independent reference translations are available. Quality metrics are therefore not calculated.")

if condition_col:
    st.subheader("Workflow / condition comparison")
    metrics = [column for column in ["edit_ratio", "time_spent_sec", "edits_per_minute", "reference_chrf", "reference_ter"] if column in analysis_df.columns]
    grouped = analysis_df.groupby(condition_col, dropna=False)[metrics].agg(["count", "mean", "median"]).round(3)
    st.dataframe(grouped, use_container_width=True)

st.subheader("Reproducible research export")
export_columns = [
    "task_type", "task_label", "analytics_validation", "analytics_version", "mt_token_count",
    "output_token_count", "insertions", "deletions", "replacements", "unchanged_tokens",
    "edit_operations", "edit_ratio", "unchanged_ratio", "length_ratio", "time_spent_sec",
    "edits_per_minute", "reference_chrf", "reference_ter",
]
export_columns = [column for column in export_columns if column in analysis_df.columns]
export_df = analysis_df[export_columns].copy()
metadata = {
    "analytics_version": ANALYTICS_VERSION,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "records_selected": len(analysis_df),
    "valid_records": valid_count,
    "source_field": source_col,
    "machine_translation_field": mt_col,
    "student_output_field": output_col,
    "reference_field": reference_col,
    "condition_field": condition_col,
    "interpretation": {
        "mt_to_post_edit": "editing effort only",
        "reference_to_output": "reference-based quality indicators",
    },
}
c1, c2 = st.columns(2)
c1.download_button("Download analysed CSV", export_df.to_csv(index=False).encode("utf-8-sig"), "eduapp_advanced_analytics.csv", "text/csv")
c2.download_button("Download analysis metadata", json.dumps(metadata, indent=2).encode("utf-8"), "eduapp_advanced_analytics_metadata.json", "application/json")
st.dataframe(export_df.head(100), use_container_width=True, hide_index=True)
