"""Task-aware dashboard for translation and post-editing submissions."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from supabase import create_client

from modules.auth import require_teacher_access
from modules.task_mode import POST_EDITING, TRANSLATION, normalize_task_type, task_type_label


st.title("Translation and Post-Editing Dashboard")
st.write(
    "Review Supabase submissions, task-specific metrics, teacher annotations, "
    "and teacher scores."
)

if not require_teacher_access("dashboard"):
    st.stop()


@st.cache_resource
def get_supabase_client():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        st.error(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit Secrets."
        )
        st.stop()


supabase = get_supabase_client()


def load_submissions() -> pd.DataFrame:
    try:
        response = (
            supabase.table("submissions")
            .select("*")
            .order("submitted_at", desc=True)
            .execute()
        )
        return pd.DataFrame(response.data or [])
    except Exception as error:
        st.error("Could not load submissions from Supabase.")
        st.code(str(error))
        return pd.DataFrame()


def load_annotations() -> pd.DataFrame:
    try:
        response = (
            supabase.table("teacher_annotations")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return pd.DataFrame(response.data or [])
    except Exception as error:
        st.warning("Could not load teacher annotations from Supabase.")
        st.code(str(error))
        return pd.DataFrame()


def numeric_mean(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return None if values.empty else values.mean()


def metric_text(value, digits=2):
    if value is None or pd.isna(value):
        return "N/A"
    return round(float(value), digits)


submissions_df = load_submissions()
annotations_df = load_annotations()

if submissions_df.empty:
    st.warning("No student submissions are available yet.")
    st.stop()

submissions_df = submissions_df.copy()
if "task_type" not in submissions_df.columns:
    submissions_df["task_type"] = POST_EDITING
submissions_df["task_type"] = submissions_df["task_type"].apply(normalize_task_type)
submissions_df["task_label"] = submissions_df["task_type"].apply(task_type_label)


st.subheader("Filter Data")
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    assignment_titles = []
    if "assignment_title" in submissions_df.columns:
        assignment_titles = sorted(
            submissions_df["assignment_title"].dropna().astype(str).unique().tolist()
        )
    selected_assignment = st.selectbox(
        "Assignment",
        ["All assignments"] + assignment_titles,
    )

with filter_col2:
    selected_task = st.selectbox(
        "Task type",
        ["All task types", "Translation", "Post-editing"],
    )

if selected_assignment != "All assignments":
    submissions_df = submissions_df[
        submissions_df["assignment_title"].astype(str) == selected_assignment
    ].copy()

if selected_task != "All task types":
    wanted_task = TRANSLATION if selected_task == "Translation" else POST_EDITING
    submissions_df = submissions_df[submissions_df["task_type"] == wanted_task].copy()

if submissions_df.empty:
    st.warning("No submissions match the selected filters.")
    st.stop()

# Keep annotations aligned with the filtered submission set.
if (
    not annotations_df.empty
    and "submission_id" in annotations_df.columns
    and "submission_id" in submissions_df.columns
):
    visible_submission_ids = set(submissions_df["submission_id"].astype(str))
    annotations_df = annotations_df[
        annotations_df["submission_id"].astype(str).isin(visible_submission_ids)
    ].copy()

translation_df = submissions_df[submissions_df["task_type"] == TRANSLATION].copy()
postedit_df = submissions_df[submissions_df["task_type"] == POST_EDITING].copy()


st.divider()
st.subheader("Summary Statistics")

summary_cols = st.columns(5)
summary_cols[0].metric("Submissions", len(submissions_df))
summary_cols[1].metric(
    "Unique students",
    submissions_df["student_id"].nunique() if "student_id" in submissions_df.columns else 0,
)
summary_cols[2].metric("Translation tasks", len(translation_df))
summary_cols[3].metric("Post-editing tasks", len(postedit_df))
summary_cols[4].metric(
    "Annotated submissions",
    (
        annotations_df["submission_id"].nunique()
        if not annotations_df.empty and "submission_id" in annotations_df.columns
        else 0
    ),
)

metric_cols = st.columns(3)
metric_cols[0].metric(
    "Average student-output words",
    metric_text(numeric_mean(submissions_df, "pe_word_count"), 1),
)
metric_cols[1].metric(
    "Average PE edit-distance ratio",
    metric_text(numeric_mean(postedit_df, "mt_pe_edit_distance_ratio"), 3),
)
metric_cols[2].metric(
    "Average MT–PE cosine similarity",
    metric_text(numeric_mean(postedit_df, "mt_pe_cosine_similarity"), 3),
)

if translation_df.empty and not postedit_df.empty:
    st.caption("The effort metrics above are calculated from post-editing records only.")
elif not translation_df.empty and postedit_df.empty:
    st.info(
        "This filtered dataset contains translation tasks only. MT–PE effort metrics "
        "are therefore not applicable."
    )
elif not translation_df.empty and not postedit_df.empty:
    st.caption(
        "Translation and post-editing records are kept separate: MT–PE effort metrics "
        "use post-editing records only."
    )


st.divider()
st.subheader("Full Submissions Dataset")
st.dataframe(submissions_df, use_container_width=True, hide_index=True)


st.divider()
st.subheader("Post-Editing Effort Metrics")

if postedit_df.empty:
    st.info("No post-editing records match the current filters.")
else:
    effort_columns = [
        "submitted_at",
        "assignment_title",
        "student_id",
        "student_name",
        "task_type",
        "source_word_count",
        "mt_word_count",
        "pe_word_count",
        "raw_mt_word_count",
        "mt_pe_word_count_difference",
        "inserted_words",
        "deleted_words",
        "replaced_segments",
        "unchanged_words",
        "mt_pe_inserted_words",
        "mt_pe_deleted_words",
        "mt_pe_replaced_words",
        "mt_pe_unchanged_words",
        "mt_pe_cosine_similarity",
        "mt_pe_edit_distance_ratio",
        "mt_pe_length_ratio",
        "mt_pe_lexical_similarity",
        "mt_pe_change_ratio",
        "mt_pe_overlap_bleu",
        "mt_pe_overlap_chrf",
        "mt_pe_overlap_ter",
        "quality_warnings",
        "mt_pe_interpretation",
    ]
    available_effort_columns = [
        column for column in effort_columns if column in postedit_df.columns
    ]
    if available_effort_columns:
        st.dataframe(
            postedit_df[available_effort_columns],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No post-editing metric columns are available yet.")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if {"student_id", "mt_pe_edit_distance_ratio"}.issubset(postedit_df.columns):
            chart_df = postedit_df[["student_id", "mt_pe_edit_distance_ratio"]].copy()
            chart_df["mt_pe_edit_distance_ratio"] = pd.to_numeric(
                chart_df["mt_pe_edit_distance_ratio"], errors="coerce"
            )
            chart_df = chart_df.dropna()
            if not chart_df.empty:
                st.markdown("**Edit-distance ratio by student**")
                st.bar_chart(chart_df.set_index("student_id")["mt_pe_edit_distance_ratio"])

    with chart_col2:
        if {"student_id", "mt_pe_cosine_similarity"}.issubset(postedit_df.columns):
            chart_df = postedit_df[["student_id", "mt_pe_cosine_similarity"]].copy()
            chart_df["mt_pe_cosine_similarity"] = pd.to_numeric(
                chart_df["mt_pe_cosine_similarity"], errors="coerce"
            )
            chart_df = chart_df.dropna()
            if not chart_df.empty:
                st.markdown("**MT–PE cosine similarity by student**")
                st.bar_chart(chart_df.set_index("student_id")["mt_pe_cosine_similarity"])


st.divider()
st.subheader("Reference-Based Quality Metrics")

quality_columns = [
    "submitted_at",
    "assignment_title",
    "student_id",
    "student_name",
    "task_type",
    "pe_reference_bleu",
    "pe_reference_chrf",
    "pe_reference_ter",
    "pe_reference_bertscore_f1",
    "raw_mt_quality_bleu",
    "raw_mt_quality_chrf",
    "raw_mt_quality_ter",
    "pe_quality_bleu",
    "pe_quality_chrf",
    "pe_quality_ter",
    "ht_quality_bleu",
    "ht_quality_chrf",
    "ht_quality_ter",
    "raw_mt_quality_bertscore_f1",
    "pe_quality_bertscore_f1",
    "ht_quality_bertscore_f1",
    "raw_mt_quality_comet",
    "pe_quality_comet",
    "ht_quality_comet",
    "advanced_metrics_status",
]
available_quality_columns = [
    column for column in quality_columns if column in submissions_df.columns
]
if available_quality_columns:
    st.dataframe(
        submissions_df[available_quality_columns],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No reference-based quality metric columns are available yet.")


st.divider()
st.subheader("Teacher Annotation Analytics")

if annotations_df.empty:
    st.info("No teacher annotations match the current filters.")
else:
    st.dataframe(annotations_df, use_container_width=True, hide_index=True)

    annotation_cols = st.columns(3)
    annotation_cols[0].metric("Total annotations", len(annotations_df))
    annotation_cols[1].metric(
        "Error categories used",
        annotations_df["category"].dropna().nunique()
        if "category" in annotations_df.columns
        else 0,
    )
    annotation_cols[2].metric(
        "Severity levels used",
        annotations_df["severity"].dropna().nunique()
        if "severity" in annotations_df.columns
        else 0,
    )

    if "category" in annotations_df.columns:
        category_counts = annotations_df["category"].dropna().value_counts()
        if not category_counts.empty:
            st.markdown("**Error-category counts**")
            st.bar_chart(category_counts)

    if "severity" in annotations_df.columns:
        severity_counts = annotations_df["severity"].dropna().value_counts()
        if not severity_counts.empty:
            st.markdown("**Severity counts**")
            st.bar_chart(severity_counts)


st.divider()
st.subheader("Teacher Review Analytics")

review_columns = [
    "submitted_at",
    "assignment_title",
    "student_id",
    "student_name",
    "task_type",
    "teacher_score",
    "teacher_feedback",
]
available_review_columns = [
    column for column in review_columns if column in submissions_df.columns
]

if "teacher_score" not in submissions_df.columns:
    st.info("No teacher-review columns are available yet.")
else:
    reviews_df = submissions_df[available_review_columns].copy()
    reviews_df["teacher_score"] = pd.to_numeric(
        reviews_df["teacher_score"], errors="coerce"
    )
    reviewed_df = reviews_df[reviews_df["teacher_score"].notna()]

    if reviewed_df.empty:
        st.info("No teacher scores are available yet.")
    else:
        st.dataframe(reviewed_df, use_container_width=True, hide_index=True)
        score_cols = st.columns(3)
        score_cols[0].metric("Reviewed submissions", len(reviewed_df))
        score_cols[1].metric("Average teacher score", round(reviewed_df["teacher_score"].mean(), 2))
        score_cols[2].metric("Highest teacher score", round(reviewed_df["teacher_score"].max(), 2))


st.divider()
st.subheader("Research Export")

csv_data = submissions_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download filtered submissions as CSV",
    data=csv_data,
    file_name="eduapp_translation_postediting_submissions.csv",
    mime="text/csv",
)

excel_output = io.BytesIO()
with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
    submissions_df.to_excel(writer, index=False, sheet_name="Submissions")
    if not annotations_df.empty:
        annotations_df.to_excel(writer, index=False, sheet_name="Annotations")
excel_output.seek(0)

st.download_button(
    "Download filtered research workbook",
    data=excel_output,
    file_name="eduapp_translation_postediting_research.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
