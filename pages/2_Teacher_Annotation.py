import pandas as pd
import streamlit as st
from supabase import create_client

from modules.taxonomy import ERROR_TAXONOMY, SEVERITY_LEVELS


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Teacher Annotation",
    page_icon="📝",
    layout="wide",
)

st.title("Teacher Annotation")
st.write("Annotate translation errors using a structured taxonomy.")


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


def load_submissions_for_annotation():
    """
    Loads student submissions from Supabase.

    This page annotates full student submissions, not separate segment rows.
    """

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


def save_annotation(annotation):
    """
    Saves a teacher annotation to Supabase.
    """

    try:
        return supabase.table("teacher_annotations").insert(annotation).execute()

    except Exception as error:
        st.error("Could not save annotation to Supabase.")
        st.write("Annotation data being sent:")
        st.json(annotation)
        st.write("Supabase error:")
        st.code(str(error))
        st.stop()


# ============================================================
# Load submissions
# ============================================================

submissions = load_submissions_for_annotation()

if submissions.empty:
    st.warning("No student submissions found. Please add a student submission first.")
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
    "Select student submission",
    submission_labels,
)

selected_submission_id = selected_label.split("ID ")[-1].strip()

selected_submission_df = submissions[
    submissions["submission_id"].astype(str) == selected_submission_id
]

if selected_submission_df.empty:
    st.error("Could not find the selected submission.")
    st.stop()

submission = selected_submission_df.iloc[0].to_dict()


# ============================================================
# Display submission details
# ============================================================

st.divider()
st.subheader("Submission Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Student ID**")
    st.write(safe_text(submission.get("student_id")))

    st.markdown("**Student Name**")
    st.write(safe_text(submission.get("student_name")))

with col2:
    st.markdown("**Assignment**")
    st.write(safe_text(submission.get("assignment_title")))

    st.markdown("**Submitted At**")
    st.write(safe_text(submission.get("submitted_at")))


st.markdown("### Source Text")
st.text_area(
    "Source text",
    value=safe_text(submission.get("source_text")),
    height=180,
    disabled=True,
)

st.markdown("### Raw Machine Translation")
st.text_area(
    "Raw MT",
    value=safe_text(submission.get("machine_translation")),
    height=180,
    disabled=True,
)

if safe_text(submission.get("reference_translation")):
    st.markdown("### Reference Translation")
    st.text_area(
        "Reference translation",
        value=safe_text(submission.get("reference_translation")),
        height=180,
        disabled=True,
    )

st.markdown("### Student Post-Edited Text")
st.text_area(
    "Post-edited text",
    value=safe_text(submission.get("post_edited_text")),
    height=250,
    disabled=True,
)


# ============================================================
# Annotation form
# ============================================================

st.divider()
st.subheader("Annotation Form")

annotation_id = st.text_input(
    "Annotation ID",
    value=f"A_{selected_submission_id}",
)

category = st.selectbox(
    "Error Category",
    list(ERROR_TAXONOMY.keys()),
)

subcategory = st.selectbox(
    "Subcategory",
    ERROR_TAXONOMY[category],
)

severity = st.selectbox(
    "Severity",
    SEVERITY_LEVELS,
)

selected_text = st.text_area(
    "Text span / phrase being annotated",
    height=80,
    placeholder="Optional: paste the specific word, phrase, or sentence you are annotating.",
)

teacher_comment = st.text_area(
    "Teacher Comment",
    height=120,
)

suggested_revision = st.text_area(
    "Suggested Revision",
    height=120,
)

annotator_id = st.text_input(
    "Annotator ID",
    value="Teacher_1",
)


if st.button("Save Annotation"):
    annotation = {
        "annotation_id": annotation_id,
        "submission_id": selected_submission_id,
        "assignment_id": safe_text(submission.get("assignment_id")),
        "assignment_title": safe_text(submission.get("assignment_title")),
        "student_id": safe_text(submission.get("student_id")),
        "student_name": safe_text(submission.get("student_name")),
        "category": category,
        "subcategory": subcategory,
        "severity": severity,
        "selected_text": selected_text,
        "teacher_comment": teacher_comment,
        "suggested_revision": suggested_revision,
        "annotator_id": annotator_id,
    }

    save_annotation(annotation)

    st.success("Annotation saved successfully.")


# ============================================================
# Existing annotations for this submission
# ============================================================

st.divider()
st.subheader("Existing Annotations for This Submission")

try:
    response = (
        supabase.table("teacher_annotations")
        .select("*")
        .eq("submission_id", selected_submission_id)
        .execute()
    )

    annotations = pd.DataFrame(response.data or [])

    if annotations.empty:
        st.info("No annotations saved for this submission yet.")
    else:
        display_columns = [
            "annotation_id",
            "category",
            "subcategory",
            "severity",
            "selected_text",
            "teacher_comment",
            "suggested_revision",
            "annotator_id",
            "created_at",
        ]

        available_columns = [
            column for column in display_columns if column in annotations.columns
        ]

        st.dataframe(
            annotations[available_columns],
            use_container_width=True,
            hide_index=True,
        )

except Exception as error:
    st.warning("Could not load existing annotations.")
    st.code(str(error))
