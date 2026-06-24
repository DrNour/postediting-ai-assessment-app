import pandas as pd
import streamlit as st
from supabase import create_client


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Post-Editing Dashboard")
st.write("Research dashboard based on Supabase submissions and teacher annotations.")


# ============================================================
# Supabase connection
# ============================================================

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


# ============================================================
# Helper functions
# ============================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def load_submissions():
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


def load_annotations():
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


def show_metric_if_available(label, value):
    try:
        if pd.isna(value):
            value = 0
    except Exception:
        pass

    st.metric(label, value)


# ============================================================
# Load data
# ============================================================

submissions_df = load_submissions()
annotations_df = load_annotations()


# ============================================================
# Main dashboard
# ============================================================

if submissions_df.empty:
    st.warning("No student submissions available yet.")
    st.stop()


# ============================================================
# Assignment filter
# ============================================================

st.subheader("Filter Data")

if "assignment_title" in submissions_df.columns:
    assignment_titles = (
        submissions_df["assignment_title"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_assignment = st.selectbox(
        "Choose assignment",
        ["All assignments"] + assignment_titles,
    )

    if selected_assignment != "All assignments":
        submissions_df = submissions_df[
            submissions_df["assignment_title"].astype(str) == selected_assignment
        ]

        if not annotations_df.empty and "assignment_title" in annotations_df.columns:
            annotations_df = annotations_df[
                annotations_df["assignment_title"].astype(str) == selected_assignment
            ]


if submissions_df.empty:
    st.warning("No submissions found for this assignment.")
    st.stop()


# ============================================================
# Summary statistics
# ============================================================

st.divider()
st.subheader("Summary Statistics")

total_submissions = len(submissions_df)

unique_students = (
    submissions_df["student_id"].nunique()
    if "student_id" in submissions_df.columns
    else 0
)

annotated_submissions = (
    annotations_df["submission_id"].nunique()
    if not annotations_df.empty and "submission_id" in annotations_df.columns
    else 0
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Submissions", total_submissions)

with col2:
    st.metric("Unique Students", unique_students)

with col3:
    st.metric("Annotated Submissions", annotated_submissions)


col4, col5, col6 = st.columns(3)

with col4:
    if "pe_word_count" in submissions_df.columns:
        avg_pe_words = submissions_df["pe_word_count"].dropna().mean()
        st.metric(
            "Average PE Word Count",
            round(avg_pe_words, 2) if pd.notna(avg_pe_words) else 0,
        )
    else:
        st.metric("Average PE Word Count", "N/A")

with col5:
    if "mt_pe_edit_distance_ratio" in submissions_df.columns:
        avg_edit_ratio = submissions_df["mt_pe_edit_distance_ratio"].dropna().mean()
        st.metric(
            "Average Edit-Distance Ratio",
            round(avg_edit_ratio, 3) if pd.notna(avg_edit_ratio) else 0,
        )
    else:
        st.metric("Average Edit-Distance Ratio", "N/A")

with col6:
    if "mt_pe_cosine_similarity" in submissions_df.columns:
        avg_cosine = submissions_df["mt_pe_cosine_similarity"].dropna().mean()
        st.metric(
            "Average MT-PE Cosine Similarity",
            round(avg_cosine, 3) if pd.notna(avg_cosine) else 0,
        )
    else:
        st.metric("Average MT-PE Cosine Similarity", "N/A")


# ============================================================
# Full dataset
# ============================================================

st.divider()
st.subheader("Full Submissions Dataset")

st.dataframe(
    submissions_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# MT vs PE metrics
# ============================================================

st.divider()
st.subheader("MT vs Post-Edited Metrics")

metric_columns = [
    "submitted_at",
    "assignment_title",
    "student_id",
    "student_name",
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
    "mt_pe_bleu",
    "mt_pe_chrf",
    "mt_pe_ter",
    "mt_pe_overlap_bleu",
    "mt_pe_overlap_chrf",
    "mt_pe_overlap_ter",
    "quality_warnings",
    "mt_pe_interpretation",
]

available_metric_columns = [
    column for column in metric_columns if column in submissions_df.columns
]

if available_metric_columns:
    st.dataframe(
        submissions_df[available_metric_columns],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No MT-PE metric columns found yet.")


# ============================================================
# Reference-based quality metrics
# ============================================================

st.divider()
st.subheader("Reference-Based Quality Metrics")

quality_columns = [
    "submitted_at",
    "assignment_title",
    "student_id",
    "student_name",
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
    st.info("No reference-based quality metric columns found yet.")


# ============================================================
# Charts: post-editing effort
# ============================================================

st.divider()
st.subheader("Post-Editing Effort Charts")

chart_cols = st.columns(2)

with chart_cols[0]:
    if "mt_pe_edit_distance_ratio" in submissions_df.columns:
        chart_df = submissions_df[["student_id", "mt_pe_edit_distance_ratio"]].dropna()
        if not chart_df.empty:
            st.markdown("**Edit-Distance Ratio by Student**")
            st.bar_chart(
                chart_df.set_index("student_id")["mt_pe_edit_distance_ratio"]
            )
        else:
            st.info("No edit-distance data available.")
    else:
        st.info("Column `mt_pe_edit_distance_ratio` not found.")

with chart_cols[1]:
    if "mt_pe_cosine_similarity" in submissions_df.columns:
        chart_df = submissions_df[["student_id", "mt_pe_cosine_similarity"]].dropna()
        if not chart_df.empty:
            st.markdown("**MT-PE Cosine Similarity by Student**")
            st.bar_chart(
                chart_df.set_index("student_id")["mt_pe_cosine_similarity"]
            )
        else:
            st.info("No cosine similarity data available.")
    else:
        st.info("Column `mt_pe_cosine_similarity` not found.")


# ============================================================
# Annotation analytics
# ============================================================

st.divider()
st.subheader("Teacher Annotation Analytics")

if annotations_df.empty:
    st.info("No teacher annotations yet.")

else:
    st.dataframe(
        annotations_df,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Annotations", len(annotations_df))

    with col2:
        if "category" in annotations_df.columns:
            st.metric(
                "Error Categories Used",
                annotations_df["category"].dropna().nunique(),
            )
        else:
            st.metric("Error Categories Used", 0)

    with col3:
        if "severity" in annotations_df.columns:
            st.metric(
                "Severity Levels Used",
                annotations_df["severity"].dropna().nunique(),
            )
        else:
            st.metric("Severity Levels Used", 0)

    if "category" in annotations_df.columns:
        category_counts = annotations_df["category"].dropna().value_counts()

        if not category_counts.empty:
            st.markdown("### Error Category Counts")
            st.bar_chart(category_counts)

    if "severity" in annotations_df.columns:
        severity_counts = annotations_df["severity"].dropna().value_counts()

        if not severity_counts.empty:
            st.markdown("### Severity Counts")
            st.bar_chart(severity_counts)

    if "subcategory" in annotations_df.columns:
        subcategory_counts = annotations_df["subcategory"].dropna().value_counts()

        if not subcategory_counts.empty:
            st.markdown("### Subcategory Counts")
            st.bar_chart(subcategory_counts)


# ============================================================
# Teacher review analytics
# ============================================================

st.divider()
st.subheader("Teacher Review Analytics")

review_columns = [
    "submitted_at",
    "assignment_title",
    "student_id",
    "student_name",
    "teacher_score",
    "teacher_feedback",
]

available_review_columns = [
    column for column in review_columns if column in submissions_df.columns
]

if not available_review_columns:
    st.info("No teacher review columns found.")

else:
    reviews_df = submissions_df[available_review_columns].copy()

    if "teacher_score" in reviews_df.columns:
        reviewed_df = reviews_df[reviews_df["teacher_score"].notna()]
    else:
        reviewed_df = pd.DataFrame()

    if reviewed_df.empty:
        st.info("No teacher score data yet.")
    else:
        st.dataframe(
            reviewed_df,
            use_container_width=True,
            hide_index=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Reviewed Submissions", len(reviewed_df))

        with col2:
            avg_score = reviewed_df["teacher_score"].dropna().mean()
            st.metric(
                "Average Teacher Score",
                round(avg_score, 2) if pd.notna(avg_score) else 0,
            )

        with col3:
            max_score = reviewed_df["teacher_score"].dropna().max()
            st.metric(
                "Highest Teacher Score",
                round(max_score, 2) if pd.notna(max_score) else 0,
            )


# ============================================================
# Research export
# ============================================================

st.divider()
st.subheader("Research Export")

csv_data = submissions_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download submissions dataset as CSV",
    data=csv_data,
    file_name="supabase_postediting_submissions.csv",
    mime="text/csv",
)

excel_buffer = pd.ExcelWriter

import io

excel_output = io.BytesIO()

with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
    submissions_df.to_excel(writer, index=False, sheet_name="Submissions")

    if not annotations_df.empty:
        annotations_df.to_excel(writer, index=False, sheet_name="Annotations")

excel_output.seek(0)

st.download_button(
    "Download research dataset as Excel",
    data=excel_output,
    file_name="supabase_postediting_research_dataset.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
