import pandas as pd
import streamlit as st
from supabase import create_client

from metrics import compare_postedit_with_raw_mt, build_research_metrics_payload


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Manual Student Submission",
    page_icon="✍️",
    layout="wide",
)

st.title("Manual Student Submission")
st.write("Manually save a student's post-edited translation to Supabase.")


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
# Helpers
# ============================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_value_for_supabase(value):
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return value

    except Exception:
        return value


def save_submission(submission):
    clean_submission = {
        key: clean_value_for_supabase(value)
        for key, value in submission.items()
    }

    try:
        return supabase.table("submissions").insert(clean_submission).execute()

    except Exception as error:
        st.error("Could not save the submission to Supabase.")
        st.write("Submission data being sent:")
        st.json(clean_submission)
        st.write("Supabase error:")
        st.code(str(error))
        st.stop()


# ============================================================
# Form
# ============================================================

st.header("Student Information")

col1, col2 = st.columns(2)

with col1:
    student_id = st.text_input("Student ID", value="S001")
    student_name = st.text_input("Student Name", value="Student A")

with col2:
    group_name = st.text_input("Group", value="Group 1")
    semester = st.text_input("Semester", value="Fall 2026")


st.header("Task Information")

col3, col4 = st.columns(2)

with col3:
    assignment_id = st.text_input("Assignment ID", value="T001")
    assignment_title = st.text_input("Assignment Title", value="University Announcement")
    domain = st.text_input("Domain", value="Institutional")

with col4:
    source_language = st.text_input("Source Language", value="Arabic")
    target_language = st.text_input("Target Language", value="English")


st.header("Translation Texts")

source_text = st.text_area(
    "Source Text",
    value="زار الوزير الجامعة أمس.",
    height=120,
)

raw_mt = st.text_area(
    "Machine Translation",
    value="The minister visited the university yesterday.",
    height=120,
)

reference_translation = st.text_area(
    "Reference Translation",
    value="The minister visited the university yesterday.",
    height=120,
)

post_edited_text = st.text_area(
    "Post-Edited Text",
    value="The minister visited the university yesterday.",
    height=120,
)

editing_time_seconds = st.number_input(
    "Editing Time in Seconds",
    min_value=0.0,
    value=52.0,
)


st.header("Research Settings")

research_mode = st.toggle(
    "Research mode",
    value=True,
    help="Stores research-ready metrics and marks advanced metrics as pending.",
)

bert_language = st.selectbox(
    "BERTScore language",
    ["en", "ar", "fr", "de", "es", "zh", "ja", "ko", "tr", "ru"],
    index=0,
)


# ============================================================
# Save
# ============================================================

if st.button("Save Submission"):
    if not safe_text(student_id):
        st.error("Please enter a student ID.")
        st.stop()

    if not safe_text(post_edited_text):
        st.error("Please enter the post-edited text.")
        st.stop()

    with st.spinner("Calculating metrics and saving submission..."):

        results = compare_postedit_with_raw_mt(
            raw_mt=raw_mt,
            post_edited_text=post_edited_text,
            human_translation=None,
            reference_text=reference_translation,
            source_text=source_text,
            teacher_score=None,
            teacher_feedback="",
            use_bert=False,
            bert_language=bert_language,
            comet_scorer=None,
        )

        submission = {
            "assignment_id": assignment_id,
            "assignment_title": assignment_title,
            "student_id": student_id,
            "student_name": student_name,
            "source_text": source_text,
            "machine_translation": raw_mt,
            "reference_translation": reference_translation,
            "post_edited_text": post_edited_text,
            "editing_time_seconds": editing_time_seconds,
            "group_name": group_name,
            "semester": semester,
            "domain": domain,
            "source_language": source_language,
            "target_language": target_language,
            "teacher_score": None,
            "teacher_feedback": "",
        }

        submission.update(
            build_research_metrics_payload(
                results,
                research_mode=research_mode,
            )
        )

        save_submission(submission)

    st.success("Submission saved successfully to Supabase.")
