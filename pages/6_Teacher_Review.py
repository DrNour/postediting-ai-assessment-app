import json

import pandas as pd
import streamlit as st
from supabase import create_client


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Teacher Review",
    page_icon="✅",
    layout="wide",
)

st.title("Teacher Review and Metric Controls")
st.write("Control research metrics and approve, edit, or reject AI-generated draft feedback.")


# ============================================================
# Teacher-only access gate
# ============================================================

configured_password = st.secrets.get("TEACHER_PASSWORD", None)

if not configured_password:
    st.error(
        "TEACHER_PASSWORD is not configured. Add it to Streamlit Secrets before using this page."
    )
    st.stop()

teacher_password = st.sidebar.text_input(
    "Teacher password",
    type="password",
)

if teacher_password != configured_password:
    st.warning("This page is restricted to the instructor.")
    st.stop()


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
# Defaults
# ============================================================

DEFAULT_METRIC_SETTINGS = {
    "research_mode": True,
    "run_advanced_metrics_now": False,

    "show_student_metrics": True,
    "show_editing_summary": True,
    "show_mt_pe_overlap_metrics": True,
    "show_reference_quality_metrics": True,
    "show_automated_interpretation": True,

    "use_bert": False,
    "bert_language": "en",
    "use_comet": False,
    "use_llm_judge": False,
}


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


def clean_value_for_supabase(value):
    try:
        if value is None:
            return None

        if isinstance(value, float) and pd.isna(value):
            return None

        if pd.isna(value):
            return None

        return value

    except Exception:
        return value


def load_metric_settings():
    settings = DEFAULT_METRIC_SETTINGS.copy()

    try:
        response = (
            supabase.table("app_metric_settings")
            .select("*")
            .eq("id", "default")
            .single()
            .execute()
        )

        if response.data:
            settings.update(response.data)

    except Exception:
        st.warning(
            "Could not load metric settings. Make sure the app_metric_settings table exists."
        )

    return settings


def save_metric_settings(settings):
    payload = {
        "id": "default",
        "research_mode": settings["research_mode"],
        "run_advanced_metrics_now": settings["run_advanced_metrics_now"],

        "show_student_metrics": settings["show_student_metrics"],
        "show_editing_summary": settings["show_editing_summary"],
        "show_mt_pe_overlap_metrics": settings["show_mt_pe_overlap_metrics"],
        "show_reference_quality_metrics": settings["show_reference_quality_metrics"],
        "show_automated_interpretation": settings["show_automated_interpretation"],

        "use_bert": settings["use_bert"],
        "bert_language": settings["bert_language"],
        "use_comet": settings["use_comet"],
        "use_llm_judge": settings["use_llm_judge"],
    }

    clean_payload = {
        key: clean_value_for_supabase(value)
        for key, value in payload.items()
    }

    try:
        supabase.table("app_metric_settings").upsert(clean_payload).execute()
        st.success("Metric settings saved successfully.")

    except Exception as error:
        st.error("Could not save metric settings.")
        st.write("Run the SQL table-creation code first in Supabase.")
        st.code(str(error))


def load_ai_feedback():
    try:
        response = (
            supabase.table("ai_feedback")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return pd.DataFrame(response.data or [])

    except Exception as error:
        st.error("Could not load AI feedback from Supabase.")
        st.code(str(error))
        return pd.DataFrame()


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


def load_feedback_with_submissions():
    feedback_df = load_ai_feedback()
    submissions_df = load_submissions()

    if feedback_df.empty:
        return pd.DataFrame()

    if submissions_df.empty:
        return feedback_df

    if "submission_id" not in feedback_df.columns:
        st.error("The ai_feedback table does not contain submission_id.")
        return feedback_df

    if "submission_id" not in submissions_df.columns:
        st.error("The submissions table does not contain submission_id.")
        return feedback_df

    merged = feedback_df.merge(
        submissions_df,
        on="submission_id",
        how="left",
        suffixes=("_feedback", "_submission"),
    )

    return merged


def save_teacher_review(review):
    clean_review = {
        key: clean_value_for_supabase(value)
        for key, value in review.items()
    }

    try:
        supabase.table("ai_feedback_reviews").insert(clean_review).execute()

        supabase.table("ai_feedback").update(
            {
                "teacher_review_status": clean_review.get("review_status"),
                "teacher_review_comment": clean_review.get("teacher_notes"),
            }
        ).eq(
            "feedback_id",
            clean_review.get("feedback_id"),
        ).execute()

    except Exception as error:
        st.error("Could not save teacher review to Supabase.")
        st.write("Review data being sent:")
        st.json(clean_review)
        st.write("Supabase error:")
        st.code(str(error))
        st.stop()


def load_teacher_reviews():
    try:
        response = (
            supabase.table("ai_feedback_reviews")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return pd.DataFrame(response.data or [])

    except Exception:
        return pd.DataFrame()


# ============================================================
# Metric controls
# ============================================================

st.header("Research Mode and Metric Controls")

metric_settings = load_metric_settings()

with st.form("metric_settings_form"):

    st.subheader("Research mode")

    research_mode = st.toggle(
        "Activate research mode",
        value=bool(metric_settings.get("research_mode", True)),
        help="When on, submissions are saved with research-ready metric fields.",
    )

    run_advanced_metrics_now = st.toggle(
        "Run advanced metrics during student submission",
        value=bool(metric_settings.get("run_advanced_metrics_now", False)),
        help=(
            "For normal class use, keep this off. "
            "For research testing, turn it on after installing required packages."
        ),
    )

    st.subheader("Student-visible feedback")

    show_student_metrics = st.checkbox(
        "Show feedback and metrics to students",
        value=bool(metric_settings.get("show_student_metrics", True)),
    )

    show_editing_summary = st.checkbox(
        "Show post-editing effort metrics",
        value=bool(metric_settings.get("show_editing_summary", True)),
    )

    show_mt_pe_overlap_metrics = st.checkbox(
        "Show MT–post-edit overlap metrics",
        value=bool(metric_settings.get("show_mt_pe_overlap_metrics", True)),
    )

    show_reference_quality_metrics = st.checkbox(
        "Show reference-based quality metrics",
        value=bool(metric_settings.get("show_reference_quality_metrics", True)),
    )

    show_automated_interpretation = st.checkbox(
        "Show automated interpretation",
        value=bool(metric_settings.get("show_automated_interpretation", True)),
    )

    st.subheader("Advanced metric activation")

    use_bert = st.checkbox(
        "Calculate BERTScore",
        value=bool(metric_settings.get("use_bert", False)),
        help="Requires bert-score to be installed.",
    )

    languages = ["en", "ar", "fr", "de", "es", "zh", "ja", "ko", "tr", "ru"]
    current_language = metric_settings.get("bert_language", "en")

    if current_language not in languages:
        current_language = "en"

    bert_language = st.selectbox(
        "BERTScore language",
        languages,
        index=languages.index(current_language),
    )

    use_comet = st.checkbox(
        "Calculate COMET",
        value=bool(metric_settings.get("use_comet", False)),
        help="The toggle is saved, but COMET also needs a scorer implementation.",
    )

    use_llm_judge = st.checkbox(
        "Use LLM judge",
        value=bool(metric_settings.get("use_llm_judge", False)),
        help="The toggle is saved, but LLM scoring needs a separate function.",
    )

    activate_all = st.checkbox(
        "Activate all available research settings",
        value=False,
        help="Turns on research mode, student feedback, BERTScore, COMET, and LLM flags.",
    )

    submitted_settings = st.form_submit_button("Save Metric Settings")

    if submitted_settings:

        if activate_all:
            research_mode = True
            run_advanced_metrics_now = True
            show_student_metrics = True
            show_editing_summary = True
            show_mt_pe_overlap_metrics = True
            show_reference_quality_metrics = True
            show_automated_interpretation = True
            use_bert = True
            use_comet = True
            use_llm_judge = True

        new_settings = {
            "research_mode": research_mode,
            "run_advanced_metrics_now": run_advanced_metrics_now,

            "show_student_metrics": show_student_metrics,
            "show_editing_summary": show_editing_summary,
            "show_mt_pe_overlap_metrics": show_mt_pe_overlap_metrics,
            "show_reference_quality_metrics": show_reference_quality_metrics,
            "show_automated_interpretation": show_automated_interpretation,

            "use_bert": use_bert,
            "bert_language": bert_language,
            "use_comet": use_comet,
            "use_llm_judge": use_llm_judge,
        }

        save_metric_settings(new_settings)


st.divider()


# ============================================================
# Load AI feedback data
# ============================================================

st.header("Teacher Review of AI Feedback")

ai_df = load_feedback_with_submissions()

if ai_df.empty:
    st.info("No AI feedback has been saved yet. Metric controls above are still available.")
    st.stop()


# ============================================================
# Select AI feedback
# ============================================================

st.subheader("Select AI Feedback")

feedback_options = ai_df["feedback_id"].dropna().astype(str).tolist()

if not feedback_options:
    st.warning("No valid feedback IDs found.")
    st.stop()

selected_feedback_id = st.selectbox(
    "Feedback ID",
    feedback_options,
)

selected_df = ai_df[ai_df["feedback_id"].astype(str) == selected_feedback_id]

if selected_df.empty:
    st.error("Could not find selected AI feedback.")
    st.stop()

selected_row = selected_df.iloc[0].to_dict()


# ============================================================
# Translation submission
# ============================================================

st.divider()
st.subheader("Translation Submission")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Source Text**")
    st.write(safe_text(selected_row.get("source_text")))

with col2:
    st.markdown("**Machine Translation**")
    st.write(safe_text(selected_row.get("machine_translation")))

with col3:
    st.markdown("**Post-Edited Text**")
    st.write(safe_text(selected_row.get("post_edited_text")))


st.markdown("### Submission Metadata")

meta_col1, meta_col2, meta_col3 = st.columns(3)

with meta_col1:
    st.write(
        f"**Student ID:** "
        f"{safe_text(selected_row.get('student_id_feedback') or selected_row.get('student_id'))}"
    )
    st.write(
        f"**Student Name:** "
        f"{safe_text(selected_row.get('student_name_feedback') or selected_row.get('student_name'))}"
    )

with meta_col2:
    st.write(
        f"**Assignment:** "
        f"{safe_text(selected_row.get('assignment_title_feedback') or selected_row.get('assignment_title'))}"
    )
    st.write(f"**Submission ID:** {safe_text(selected_row.get('submission_id'))}")

with meta_col3:
    st.write(f"**Model:** {safe_text(selected_row.get('model_name'))}")
    st.write(
        f"**Feedback created at:** "
        f"{safe_text(selected_row.get('created_at_feedback') or selected_row.get('created_at'))}"
    )


# ============================================================
# AI draft feedback
# ============================================================

st.divider()
st.subheader("AI Draft Feedback")

st.markdown("**AI Overall Comment**")
ai_overall_comment = safe_text(selected_row.get("ai_overall_comment"))
st.write(ai_overall_comment)

st.markdown("**AI Risk Level**")
risk_level = safe_text(selected_row.get("ai_risk_level")).lower()

if risk_level == "low":
    st.success("Low risk")
elif risk_level == "medium":
    st.warning("Medium risk")
elif risk_level == "high":
    st.error("High risk")
else:
    st.info(risk_level or "No risk level recorded.")

st.markdown("**AI Risk Warnings**")
risk_warnings = parse_json_value(selected_row.get("ai_risk_warnings"))

if risk_warnings:
    st.json(risk_warnings)
else:
    st.info("No risk warnings recorded.")

st.markdown("**AI Possible Errors**")
possible_errors = parse_json_value(selected_row.get("ai_possible_errors"))

if possible_errors:
    st.json(possible_errors)
else:
    st.info("No possible errors recorded.")

st.markdown("**AI Rubric Scores**")
rubric_scores = parse_json_value(selected_row.get("ai_rubric_scores"))

if rubric_scores:
    st.json(rubric_scores)
else:
    st.info("No rubric scores recorded.")

with st.expander("Raw AI JSON"):
    raw_json = parse_json_value(selected_row.get("ai_raw_json"))
    st.json(raw_json)


# ============================================================
# Teacher decision
# ============================================================

st.divider()
st.subheader("Teacher Decision")

review_status = st.selectbox(
    "Review Status",
    ["approved", "edited", "rejected", "needs_discussion"],
)

teacher_final_feedback = st.text_area(
    "Teacher Final Feedback",
    value=ai_overall_comment,
    height=180,
)

teacher_notes = st.text_area(
    "Teacher Notes",
    placeholder="Explain why you approved, edited, or rejected the AI feedback.",
    height=120,
)

usefulness_rating = st.slider(
    "How useful was the AI feedback?",
    min_value=1,
    max_value=5,
    value=3,
)

teacher_review_time_seconds = st.number_input(
    "Teacher Review Time in Seconds",
    min_value=0.0,
    value=0.0,
)

reviewer_id = st.text_input(
    "Reviewer ID",
    value="Teacher_1",
)

review_id = st.text_input(
    "Review ID",
    value=f"RV_{selected_feedback_id}",
)


if st.button("Save Teacher Review"):
    review = {
        "review_id": review_id,
        "feedback_id": selected_feedback_id,
        "submission_id": safe_text(selected_row.get("submission_id")),
        "assignment_id": safe_text(
            selected_row.get("assignment_id_feedback")
            or selected_row.get("assignment_id")
        ),
        "assignment_title": safe_text(
            selected_row.get("assignment_title_feedback")
            or selected_row.get("assignment_title")
        ),
        "student_id": safe_text(
            selected_row.get("student_id_feedback")
            or selected_row.get("student_id")
        ),
        "student_name": safe_text(
            selected_row.get("student_name_feedback")
            or selected_row.get("student_name")
        ),
        "review_status": review_status,
        "teacher_final_feedback": teacher_final_feedback,
        "teacher_notes": teacher_notes,
        "reviewer_id": reviewer_id,
        "usefulness_rating": usefulness_rating,
        "teacher_review_time_seconds": teacher_review_time_seconds,
    }

    save_teacher_review(review)

    st.success("Teacher review saved successfully.")


# ============================================================
# Saved reviews
# ============================================================

st.divider()
st.subheader("Saved Teacher Reviews")

reviews_df = load_teacher_reviews()

if reviews_df.empty:
    st.info("No teacher reviews saved yet.")

else:
    st.dataframe(
        reviews_df,
        use_container_width=True,
        hide_index=True,
    )
