import json

import pandas as pd
import streamlit as st
from supabase import create_client

from modules.ai_feedback import check_feedback_risk, generate_ai_feedback
from metrics import compare_mt_pe

try:
    from modules.similarity import semantic_similarity
except Exception:
    semantic_similarity = None


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Feedback",
    page_icon="💬",
    layout="wide",
)

st.title("AI Feedback Generator")
st.write("Generate draft AI feedback for a post-edited translation. Teacher review is required.")

st.warning(
    "AI-generated feedback is a draft and must be reviewed by a teacher before assessment use."
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


def clean_value_for_supabase(value):
    try:
        if value is None:
            return None

        if isinstance(value, float):
            if pd.isna(value):
                return None

        if pd.isna(value):
            return None

        return value

    except Exception:
        return value


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


def save_ai_feedback_to_supabase(feedback_record):
    clean_record = {
        key: clean_value_for_supabase(value)
        for key, value in feedback_record.items()
    }

    try:
        return supabase.table("ai_feedback").insert(clean_record).execute()

    except Exception as error:
        st.error("Could not save AI feedback to Supabase.")
        st.write("Feedback data being sent:")
        st.json(clean_record)
        st.write("Supabase error:")
        st.code(str(error))
        st.stop()


def load_existing_feedback(submission_id):
    try:
        response = (
            supabase.table("ai_feedback")
            .select("*")
            .eq("submission_id", submission_id)
            .order("created_at", desc=True)
            .execute()
        )

        return pd.DataFrame(response.data or [])

    except Exception:
        return pd.DataFrame()


# ============================================================
# Load submissions
# ============================================================

submissions = load_submissions()

if submissions.empty:
    st.warning("No student submissions found. Please add student submissions first.")
    st.stop()


# ============================================================
# Assignment filter
# ============================================================

st.subheader("Choose Submission")

if "assignment_title" in submissions.columns:
    assignment_titles = (
        submissions["assignment_title"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_assignment = st.selectbox(
        "Filter by assignment",
        ["All assignments"] + assignment_titles,
    )

    if selected_assignment != "All assignments":
        submissions = submissions[
            submissions["assignment_title"].astype(str) == selected_assignment
        ]

if submissions.empty:
    st.warning("No submissions found for this assignment.")
    st.stop()


# ============================================================
# Select submission
# ============================================================

submission_labels = []

for _, row in submissions.iterrows():
    label = (
        f"{safe_text(row.get('student_id'))} — "
        f"{safe_text(row.get('student_name'))} — "
        f"{safe_text(row.get('assignment_title'))} — "
        f"ID {safe_text(row.get('submission_id'))}"
    )
    submission_labels.append(label)

selected_label = st.selectbox(
    "Select Submission",
    submission_labels,
)

selected_submission_id = selected_label.split("ID ")[-1].strip()

selected_df = submissions[
    submissions["submission_id"].astype(str) == selected_submission_id
]

if selected_df.empty:
    st.error("Could not find the selected submission.")
    st.stop()

submission = selected_df.iloc[0].to_dict()


# ============================================================
# Display submission details
# ============================================================

st.subheader("Submission Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Source Text**")
    st.write(safe_text(submission.get("source_text")))

with col2:
    st.markdown("**Machine Translation**")
    st.write(safe_text(submission.get("machine_translation")))

with col3:
    st.markdown("**Post-Edited Text**")
    st.write(safe_text(submission.get("post_edited_text")))


# ============================================================
# Automatic metrics
# ============================================================

st.divider()
st.subheader("Automatic Metrics")

metrics = compare_mt_pe(
    submission.get("machine_translation"),
    submission.get("post_edited_text"),
)

st.json(metrics)

mt_pe_semantic = None
source_pe_semantic = None

if semantic_similarity is not None:
    with st.spinner("Calculating semantic similarity..."):
        mt_pe_semantic = semantic_similarity(
            safe_text(submission.get("machine_translation")),
            safe_text(submission.get("post_edited_text")),
        )

        source_pe_semantic = semantic_similarity(
            safe_text(submission.get("source_text")),
            safe_text(submission.get("post_edited_text")),
        )

    col4, col5 = st.columns(2)

    with col4:
        st.metric("MT vs PE Semantic Similarity", mt_pe_semantic)

    with col5:
        st.metric("Source vs PE Semantic Similarity", source_pe_semantic)

else:
    st.info("Semantic similarity module not available.")


# ============================================================
# Generate AI feedback
# ============================================================

st.divider()
st.subheader("Generate AI Feedback")

model_name = st.text_input(
    "Model name",
    value="gpt-4.1-mini",
)

editing_time_seconds = submission.get("editing_time_seconds")

if editing_time_seconds is None:
    editing_time_seconds = 0

if st.button("Generate AI Feedback"):
    with st.spinner("Generating AI feedback..."):
        result = generate_ai_feedback(
            source_text=safe_text(submission.get("source_text")),
            mt_output=safe_text(submission.get("machine_translation")),
            post_edited_text=safe_text(submission.get("post_edited_text")),
            editing_time_seconds=editing_time_seconds,
            lexical_similarity=metrics.get("lexical_similarity"),
            change_ratio=metrics.get("change_ratio"),
            mt_pe_semantic_similarity=mt_pe_semantic,
            source_pe_semantic_similarity=source_pe_semantic,
            model_name=model_name,
        )

    if result["success"]:
        feedback = result["feedback"]

        st.session_state["latest_feedback"] = feedback
        st.session_state["latest_feedback_submission_id"] = selected_submission_id
        st.session_state["latest_feedback_model_name"] = model_name

        st.success("AI feedback generated.")

        st.subheader("Overall Comment")
        st.write(feedback.get("overall_comment", ""))

        st.subheader("Possible Errors")
        st.json(feedback.get("possible_errors", []))

        st.subheader("Rubric Scores")
        st.json(feedback.get("rubric_scores", {}))

        risk = check_feedback_risk(feedback)

        st.session_state["latest_feedback_risk"] = risk

        st.subheader("Risk / Hallucination Warning")
        st.write(f"Risk level: **{risk['risk_level']}**")

        if risk["warnings"]:
            for warning in risk["warnings"]:
                st.warning(warning)
        else:
            st.success("No major warning detected.")

        st.subheader("Raw JSON")
        st.code(
            json.dumps(feedback, ensure_ascii=False, indent=2),
            language="json",
        )

    else:
        st.error(result["error"])
        st.subheader("Raw Model Output")
        st.code(result.get("raw_output", ""))


# ============================================================
# Save feedback
# ============================================================

if (
    "latest_feedback" in st.session_state
    and st.session_state.get("latest_feedback_submission_id") == selected_submission_id
):
    feedback = st.session_state["latest_feedback"]
    risk = st.session_state.get(
        "latest_feedback_risk",
        check_feedback_risk(feedback),
    )

    feedback_id = st.text_input(
        "Feedback ID",
        value=f"FB_{selected_submission_id}",
    )

    if st.button("Save AI Feedback for Teacher Review"):
        feedback_record = {
            "feedback_id": feedback_id,
            "submission_id": selected_submission_id,
            "assignment_id": safe_text(submission.get("assignment_id")),
            "assignment_title": safe_text(submission.get("assignment_title")),
            "student_id": safe_text(submission.get("student_id")),
            "student_name": safe_text(submission.get("student_name")),
            "model_name": st.session_state.get("latest_feedback_model_name", model_name),
            "ai_overall_comment": feedback.get("overall_comment", ""),
            "ai_possible_errors": feedback.get("possible_errors", []),
            "ai_rubric_scores": feedback.get("rubric_scores", {}),
            "ai_raw_json": feedback,
            "ai_risk_level": risk["risk_level"],
            "ai_risk_warnings": risk.get("warnings", []),
            "teacher_review_status": "pending",
        }

        save_ai_feedback_to_supabase(feedback_record)

        st.success("AI feedback saved for teacher review.")


# ============================================================
# Existing AI feedback
# ============================================================

st.divider()
st.subheader("Existing AI Feedback for This Submission")

existing_feedback = load_existing_feedback(selected_submission_id)

if existing_feedback.empty:
    st.info("No AI feedback saved for this submission yet.")

else:
    display_columns = [
        "created_at",
        "feedback_id",
        "model_name",
        "ai_overall_comment",
        "ai_risk_level",
        "teacher_review_status",
    ]

    available_columns = [
        column for column in display_columns if column in existing_feedback.columns
    ]

    st.dataframe(
        existing_feedback[available_columns],
        use_container_width=True,
        hide_index=True,
    )
