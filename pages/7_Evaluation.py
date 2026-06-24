import io
import json

import pandas as pd
import streamlit as st
from supabase import create_client


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Research Evaluation",
    page_icon="📈",
    layout="wide",
)

st.title("Research Evaluation and Export")
st.write(
    "Prepare research-ready evaluation data from student submissions, "
    "teacher annotations, AI feedback, and teacher review decisions."
)


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


def parse_json_value(value):
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(value)
    except Exception:
        return value


def load_table(table_name, order_column="created_at"):
    try:
        query = supabase.table(table_name).select("*")

        if order_column:
            query = query.order(order_column, desc=True)

        response = query.execute()

        return pd.DataFrame(response.data or [])

    except Exception as error:
        st.warning(f"Could not load `{table_name}` from Supabase.")
        st.code(str(error))
        return pd.DataFrame()


def extract_ai_first_category(value):
    """
    Extracts the first AI-predicted error category from ai_possible_errors.
    Works with list/dict/string JSON formats.
    """

    parsed = parse_json_value(value)

    if not parsed:
        return None

    if isinstance(parsed, list) and len(parsed) > 0:
        first_item = parsed[0]

        if isinstance(first_item, dict):
            return (
                first_item.get("category")
                or first_item.get("error_category")
                or first_item.get("type")
            )

        return str(first_item)

    if isinstance(parsed, dict):
        return (
            parsed.get("category")
            or parsed.get("error_category")
            or parsed.get("type")
        )

    return None


def extract_ai_confidence(value):
    """
    Extracts confidence if available from ai_possible_errors.
    """

    parsed = parse_json_value(value)

    if not parsed:
        return None

    if isinstance(parsed, list) and len(parsed) > 0:
        first_item = parsed[0]

        if isinstance(first_item, dict):
            return (
                first_item.get("confidence")
                or first_item.get("score")
                or first_item.get("probability")
            )

    if isinstance(parsed, dict):
        return (
            parsed.get("confidence")
            or parsed.get("score")
            or parsed.get("probability")
        )

    return None


def build_evaluation_dataset():
    """
    Builds a research-ready dataframe from:

    submissions
    teacher_annotations
    ai_feedback
    ai_feedback_reviews
    """

    submissions = load_table("submissions", order_column="submitted_at")
    annotations = load_table("teacher_annotations", order_column="created_at")
    ai_feedback = load_table("ai_feedback", order_column="created_at")
    ai_reviews = load_table("ai_feedback_reviews", order_column="created_at")

    if submissions.empty:
        return pd.DataFrame(), submissions, annotations, ai_feedback, ai_reviews

    eval_df = submissions.copy()

    # --------------------------------------------------------
    # Merge teacher annotations
    # --------------------------------------------------------

    if not annotations.empty and "submission_id" in annotations.columns:
        annotation_summary = (
            annotations
            .groupby("submission_id")
            .agg(
                teacher_annotation_count=("annotation_id", "count"),
                teacher_category=("category", lambda x: first_non_empty(x)),
                teacher_subcategory=("subcategory", lambda x: first_non_empty(x)),
                teacher_severity=("severity", lambda x: first_non_empty(x)),
                teacher_annotation_comments=("teacher_comment", join_non_empty),
                teacher_suggested_revisions=("suggested_revision", join_non_empty),
            )
            .reset_index()
        )

        eval_df = eval_df.merge(
            annotation_summary,
            on="submission_id",
            how="left",
        )

    else:
        eval_df["teacher_annotation_count"] = 0
        eval_df["teacher_category"] = None
        eval_df["teacher_subcategory"] = None
        eval_df["teacher_severity"] = None
        eval_df["teacher_annotation_comments"] = None
        eval_df["teacher_suggested_revisions"] = None

    # --------------------------------------------------------
    # Merge AI feedback
    # --------------------------------------------------------

    if not ai_feedback.empty and "submission_id" in ai_feedback.columns:
        ai_feedback = ai_feedback.copy()

        ai_feedback["ai_category"] = ai_feedback["ai_possible_errors"].apply(
            extract_ai_first_category
        )

        ai_feedback["ai_confidence"] = ai_feedback["ai_possible_errors"].apply(
            extract_ai_confidence
        )

        ai_summary = (
            ai_feedback
            .groupby("submission_id")
            .agg(
                ai_feedback_count=("feedback_id", "count"),
                feedback_id=("feedback_id", lambda x: first_non_empty(x)),
                model_name=("model_name", lambda x: first_non_empty(x)),
                ai_overall_comment=("ai_overall_comment", lambda x: first_non_empty(x)),
                ai_risk_level=("ai_risk_level", lambda x: first_non_empty(x)),
                ai_category=("ai_category", lambda x: first_non_empty(x)),
                ai_confidence=("ai_confidence", lambda x: first_non_empty(x)),
                teacher_review_status_from_feedback=(
                    "teacher_review_status",
                    lambda x: first_non_empty(x),
                ),
            )
            .reset_index()
        )

        eval_df = eval_df.merge(
            ai_summary,
            on="submission_id",
            how="left",
        )

    else:
        eval_df["ai_feedback_count"] = 0
        eval_df["feedback_id"] = None
        eval_df["model_name"] = None
        eval_df["ai_overall_comment"] = None
        eval_df["ai_risk_level"] = None
        eval_df["ai_category"] = None
        eval_df["ai_confidence"] = None
        eval_df["teacher_review_status_from_feedback"] = None

    # --------------------------------------------------------
    # Merge AI feedback reviews
    # --------------------------------------------------------

    if not ai_reviews.empty and "submission_id" in ai_reviews.columns:
        review_summary = (
            ai_reviews
            .groupby("submission_id")
            .agg(
                ai_review_count=("review_id", "count"),
                review_id=("review_id", lambda x: first_non_empty(x)),
                review_status=("review_status", lambda x: first_non_empty(x)),
                teacher_final_feedback=(
                    "teacher_final_feedback",
                    lambda x: first_non_empty(x),
                ),
                teacher_notes=("teacher_notes", lambda x: first_non_empty(x)),
                reviewer_id=("reviewer_id", lambda x: first_non_empty(x)),
                usefulness_rating=("usefulness_rating", lambda x: first_non_empty(x)),
                teacher_review_time_seconds=(
                    "teacher_review_time_seconds",
                    lambda x: first_non_empty(x),
                ),
            )
            .reset_index()
        )

        eval_df = eval_df.merge(
            review_summary,
            on="submission_id",
            how="left",
        )

    else:
        eval_df["ai_review_count"] = 0
        eval_df["review_id"] = None
        eval_df["review_status"] = None
        eval_df["teacher_final_feedback"] = None
        eval_df["teacher_notes"] = None
        eval_df["reviewer_id"] = None
        eval_df["usefulness_rating"] = None
        eval_df["teacher_review_time_seconds"] = None

    return eval_df, submissions, annotations, ai_feedback, ai_reviews


def first_non_empty(series):
    for value in series:
        if value is not None and str(value).strip() != "":
            return value
    return None


def join_non_empty(series):
    values = [
        str(value).strip()
        for value in series
        if value is not None and str(value).strip() != ""
    ]

    if not values:
        return None

    return " | ".join(values)


def calculate_review_status_counts(eval_df):
    if "review_status" not in eval_df.columns:
        return pd.DataFrame()

    counts = (
        eval_df["review_status"]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )

    counts.columns = ["review_status", "count"]

    return counts


def calculate_ai_teacher_agreement(eval_df):
    """
    Calculates simple agreement and Cohen's kappa between teacher_category
    and ai_category, when available.
    """

    required_columns = ["teacher_category", "ai_category"]

    for column in required_columns:
        if column not in eval_df.columns:
            return {
                "error": f"Missing column: {column}"
            }

    comparison_df = eval_df[
        ["submission_id", "teacher_category", "ai_category", "ai_confidence",
         "review_status", "ai_risk_level"]
    ].copy()

    comparison_df = comparison_df.dropna(subset=["teacher_category", "ai_category"])

    comparison_df = comparison_df[
        (comparison_df["teacher_category"].astype(str).str.strip() != "")
        & (comparison_df["ai_category"].astype(str).str.strip() != "")
    ]

    if comparison_df.empty:
        return {
            "error": "No comparable AI and teacher categories available yet."
        }

    try:
        from sklearn.metrics import classification_report, cohen_kappa_score

        teacher_labels = comparison_df["teacher_category"].astype(str)
        ai_labels = comparison_df["ai_category"].astype(str)

        kappa = cohen_kappa_score(teacher_labels, ai_labels)

        report = classification_report(
            teacher_labels,
            ai_labels,
            output_dict=True,
            zero_division=0,
        )

        return {
            "kappa": kappa,
            "classification_report": report,
            "comparison_df": comparison_df,
        }

    except Exception as error:
        return {
            "error": str(error)
        }


def summarise_usefulness(eval_df):
    if "usefulness_rating" not in eval_df.columns:
        return {
            "error": "No usefulness_rating column available."
        }

    ratings = pd.to_numeric(
        eval_df["usefulness_rating"],
        errors="coerce",
    ).dropna()

    if ratings.empty:
        return {
            "error": "No usefulness ratings available yet."
        }

    return {
        "mean": round(ratings.mean(), 3),
        "median": round(ratings.median(), 3),
        "min": int(ratings.min()),
        "max": int(ratings.max()),
        "count": int(ratings.count()),
        "std": round(ratings.std(), 3) if ratings.count() > 1 else 0,
    }


def usefulness_by_review_status(eval_df):
    if "review_status" not in eval_df.columns:
        return pd.DataFrame()

    if "usefulness_rating" not in eval_df.columns:
        return pd.DataFrame()

    temp = eval_df.copy()
    temp["usefulness_rating"] = pd.to_numeric(
        temp["usefulness_rating"],
        errors="coerce",
    )

    temp = temp.dropna(subset=["review_status", "usefulness_rating"])

    if temp.empty:
        return pd.DataFrame()

    return (
        temp
        .groupby("review_status")["usefulness_rating"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )


# ============================================================
# Build dataset
# ============================================================

eval_df, submissions_df, annotations_df, ai_feedback_df, ai_reviews_df = (
    build_evaluation_dataset()
)

if eval_df.empty:
    st.warning("No evaluation data available yet.")
    st.stop()


# ============================================================
# Dataset overview
# ============================================================

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Submissions", len(submissions_df))

with col2:
    st.metric("Teacher Annotations", len(annotations_df))

with col3:
    st.metric("AI Feedback Records", len(ai_feedback_df))

with col4:
    st.metric("AI Feedback Reviews", len(ai_reviews_df))


# ============================================================
# Evaluation dataset
# ============================================================

st.divider()
st.subheader("Evaluation Dataset")

preferred_columns = [
    "submitted_at",
    "submission_id",
    "assignment_id",
    "assignment_title",
    "student_id",
    "student_name",
    "source_text",
    "machine_translation",
    "reference_translation",
    "post_edited_text",

    "mt_pe_cosine_similarity",
    "mt_pe_edit_distance_ratio",
    "mt_pe_length_ratio",
    "mt_pe_lexical_similarity",
    "mt_pe_change_ratio",
    "mt_pe_overlap_bleu",
    "mt_pe_overlap_chrf",
    "mt_pe_overlap_ter",

    "raw_mt_quality_bleu",
    "raw_mt_quality_chrf",
    "raw_mt_quality_ter",
    "pe_quality_bleu",
    "pe_quality_chrf",
    "pe_quality_ter",
    "raw_mt_quality_bertscore_f1",
    "pe_quality_bertscore_f1",
    "raw_mt_quality_comet",
    "pe_quality_comet",
    "advanced_metrics_status",

    "teacher_annotation_count",
    "teacher_category",
    "teacher_subcategory",
    "teacher_severity",
    "teacher_annotation_comments",
    "teacher_suggested_revisions",

    "ai_feedback_count",
    "feedback_id",
    "model_name",
    "ai_overall_comment",
    "ai_risk_level",
    "ai_category",
    "ai_confidence",

    "ai_review_count",
    "review_id",
    "review_status",
    "teacher_final_feedback",
    "teacher_notes",
    "reviewer_id",
    "usefulness_rating",
    "teacher_review_time_seconds",

    "teacher_score",
    "teacher_feedback",
]

available_columns = [
    column for column in preferred_columns if column in eval_df.columns
]

remaining_columns = [
    column for column in eval_df.columns if column not in available_columns
]

display_df = eval_df[available_columns + remaining_columns]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# Download buttons
# ============================================================

csv_data = display_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Download Evaluation Dataset as CSV",
    data=csv_data,
    file_name="postediting_evaluation_dataset.csv",
    mime="text/csv",
)

excel_output = io.BytesIO()

with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
    display_df.to_excel(writer, index=False, sheet_name="Evaluation Dataset")

    if not submissions_df.empty:
        submissions_df.to_excel(writer, index=False, sheet_name="Submissions")

    if not annotations_df.empty:
        annotations_df.to_excel(writer, index=False, sheet_name="Annotations")

    if not ai_feedback_df.empty:
        ai_feedback_df.to_excel(writer, index=False, sheet_name="AI Feedback")

    if not ai_reviews_df.empty:
        ai_reviews_df.to_excel(writer, index=False, sheet_name="AI Reviews")

excel_output.seek(0)

st.download_button(
    label="Download Full Research Workbook as Excel",
    data=excel_output,
    file_name="postediting_research_workbook.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


# ============================================================
# Teacher review outcomes
# ============================================================

st.divider()
st.subheader("Teacher Review Outcomes")

review_counts = calculate_review_status_counts(eval_df)

if review_counts.empty:
    st.info("No teacher review outcomes available yet.")
else:
    st.dataframe(
        review_counts,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        review_counts.set_index("review_status")["count"]
    )


# ============================================================
# AI vs Teacher Agreement
# ============================================================

st.divider()
st.subheader("AI vs Teacher Agreement")

agreement = calculate_ai_teacher_agreement(eval_df)

if "error" in agreement:
    st.warning(agreement["error"])
else:
    st.metric("Cohen's Kappa", round(agreement["kappa"], 3))

    report_df = pd.DataFrame(
        agreement["classification_report"]
    ).transpose()

    st.markdown("**Classification Report**")
    st.dataframe(
        report_df,
        use_container_width=True,
    )

    st.markdown("**Compared Cases**")

    comparison_columns = [
        "submission_id",
        "teacher_category",
        "ai_category",
        "ai_confidence",
        "review_status",
        "ai_risk_level",
    ]

    available_comparison_columns = [
        column
        for column in comparison_columns
        if column in agreement["comparison_df"].columns
    ]

    st.dataframe(
        agreement["comparison_df"][available_comparison_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Teacher usefulness ratings
# ============================================================

st.divider()
st.subheader("Teacher Usefulness Ratings")

usefulness = summarise_usefulness(eval_df)

if "error" in usefulness:
    st.info(usefulness["error"])
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Usefulness", usefulness["mean"])

    with col2:
        st.metric("Median", usefulness["median"])

    with col3:
        st.metric("Number of Ratings", usefulness["count"])

    st.json(usefulness)


# ============================================================
# Usefulness by review status
# ============================================================

st.divider()
st.subheader("Usefulness by Review Status")

usefulness_status = usefulness_by_review_status(eval_df)

if usefulness_status.empty:
    st.info("No usefulness-by-status summary available yet.")
else:
    st.dataframe(
        usefulness_status,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Dissertation-ready tables
# ============================================================

st.divider()
st.subheader("Research-Ready Tables")

st.markdown(
    """
You can use the exported workbook for:

1. **Post-editing effort analysis**  
   Edit-distance ratio, cosine similarity, lexical change ratio, inserted/deleted/replaced/unchanged words.

2. **Reference-based quality analysis**  
   BLEU, chrF, TER, optional BERTScore, optional COMET.

3. **Teacher annotation analysis**  
   Error category, subcategory, severity, comments, and suggested revisions.

4. **AI-feedback evaluation**  
   AI risk level, AI-generated categories, teacher review status, and usefulness ratings.

5. **AI-teacher agreement analysis**  
   Cohen's kappa and classification-report style summaries where comparable labels exist.
"""
)
