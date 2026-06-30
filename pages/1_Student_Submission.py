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

def load_metric_settings():
    """
    Loads instructor-controlled metric settings from Supabase.
    Students cannot change these settings.
    """

    default_settings = {
        "research_mode": True,
        "run_advanced_metrics_now": False,

        "show_student_metrics": True,
        "show_editing_summary": True,
        "show_mt_pe_overlap_metrics": True,
        "show_reference_quality_metrics": True,
        "show_automated_interpretation": True,

        "use_semantic_cosine": True,
        "use_bert": False,
        "bert_language": "en",
        "use_comet": False,
        "use_llm_judge": False,
    }

    try:
        response = (
            supabase.table("app_metric_settings")
            .select("*")
            .eq("id", "default")
            .single()
            .execute()
        )

        if response.data:
            default_settings.update(response.data)

        return default_settings

    except Exception:
        return default_settings
# ============================================================
# Instructor-controlled metric settings
# Students can SEE feedback and metrics, but they cannot change
# which metrics are calculated.
# ============================================================

INSTRUCTOR_METRIC_SETTINGS = {
    # Save research-ready fields to Supabase
    "research_mode": True,

    # Student-visible feedback sections
    "show_student_metrics": True,
    "show_editing_summary": True,
    "show_mt_pe_overlap_metrics": True,
    "show_reference_quality_metrics": True,
    "show_automated_interpretation": True,

    # Advanced optional metrics
    # Keep these False for normal class use.
    # Change to True later when you want research/article mode.
    "use_bert": False,
    "bert_language": "en",
    "use_comet": False,
}


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


def format_metric(value, digits=3):
    """
    Safely formats numeric metric values for Streamlit display.
    """
    if value is None:
        return "N/A"

    try:
        return round(float(value), digits)
    except Exception:
        return value


def display_student_feedback_and_metrics(results, settings):
    """
    Displays feedback and metrics to students.

    Students can see these results, but they cannot control which
    metrics are calculated. The instructor controls that through
    INSTRUCTOR_METRIC_SETTINGS.
    """

    if not settings["show_student_metrics"]:
        return

    st.header("Feedback and Metrics")

    # --------------------------------------------------------
    # 1. Post-editing effort summary
    # --------------------------------------------------------

    if settings["show_editing_summary"]:
        st.subheader("Post-editing effort")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Raw MT word count",
                format_metric(results.get("raw_mt_word_count"), 0),
            )

        with col2:
            st.metric(
                "Post-edited word count",
                format_metric(results.get("pe_word_count"), 0),
            )

        with col3:
            st.metric(
                "Word-count difference",
                format_metric(results.get("mt_pe_word_count_difference"), 0),
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric(
                "MT–PE similarity",
                format_metric(results.get("mt_pe_cosine_similarity"), 3),
            )

        with col5:
            st.metric(
                "Edit-distance ratio",
                format_metric(results.get("mt_pe_edit_distance_ratio"), 3),
            )

        with col6:
            st.metric(
                "Length ratio",
                format_metric(results.get("mt_pe_length_ratio"), 3),
            )

        col7, col8, col9 = st.columns(3)

        with col7:
            st.metric(
                "Inserted words",
                format_metric(results.get("mt_pe_inserted_words"), 0),
            )

        with col8:
            st.metric(
                "Deleted words",
                format_metric(results.get("mt_pe_deleted_words"), 0),
            )

        with col9:
            st.metric(
                "Replaced words",
                format_metric(results.get("mt_pe_replaced_words"), 0),
            )

    # --------------------------------------------------------
    # 2. MT-to-post-edit overlap metrics
    # --------------------------------------------------------

    if settings["show_mt_pe_overlap_metrics"]:
        st.subheader("MT–post-edit overlap")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "BLEU against raw MT",
                format_metric(results.get("mt_pe_overlap_bleu"), 2),
            )

        with col2:
            st.metric(
                "chrF against raw MT",
                format_metric(results.get("mt_pe_overlap_chrf"), 2),
            )

        with col3:
            st.metric(
                "TER against raw MT",
                format_metric(results.get("mt_pe_overlap_ter"), 2),
            )

        st.caption(
            "These metrics compare the post-edited text with the original MT output. "
            "They indicate editing overlap and effort, not final translation quality."
        )

    # --------------------------------------------------------
    # 3. Reference-based quality metrics
    # --------------------------------------------------------

    if settings["show_reference_quality_metrics"]:
        st.subheader("Reference-based quality")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "PE BLEU",
                format_metric(results.get("pe_quality_bleu"), 2),
            )

        with col2:
            st.metric(
                "PE chrF",
                format_metric(results.get("pe_quality_chrf"), 2),
            )

        with col3:
            st.metric(
                "PE TER",
                format_metric(results.get("pe_quality_ter"), 2),
            )

        st.caption(
            "These metrics are quality-oriented only when an independent reference "
            "translation is available."
        )

    # --------------------------------------------------------
    # 4. Advanced metrics
    # --------------------------------------------------------

    if settings["use_bert"]:
        st.subheader("Advanced metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Raw MT BERTScore F1",
                format_metric(results.get("raw_mt_quality_bertscore_f1"), 4),
            )

        with col2:
            st.metric(
                "PE BERTScore F1",
                format_metric(results.get("pe_quality_bertscore_f1"), 4),
            )

        with col3:
            st.metric(
                "HT BERTScore F1",
                format_metric(results.get("ht_quality_bertscore_f1"), 4),
            )

    # --------------------------------------------------------
    # 5. Automated interpretation
    # --------------------------------------------------------

    if settings["show_automated_interpretation"]:
        interpretation = results.get("mt_pe_interpretation")

        if interpretation:
            st.subheader("Automated interpretation")
            st.info(interpretation)


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
    assignment_title = st.text_input(
        "Assignment Title",
        value="University Announcement",
    )
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


# ============================================================
# Instructor note
# This is informational only. Students cannot change settings.
# You may remove this expander if you do not want students to see it.
# ============================================================

with st.expander("Metric information", expanded=False):
    st.write(
        "Metrics are calculated automatically after submission. "
        "The instructor controls which metrics are active."
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

        # ----------------------------------------------------
        # Calculate metrics using instructor-controlled settings
        # ----------------------------------------------------

        results = compare_postedit_with_raw_mt(
            raw_mt=raw_mt,
            post_edited_text=post_edited_text,
            human_translation=None,
            reference_text=reference_translation,
            source_text=source_text,
            teacher_score=None,
            teacher_feedback="",

            # Students cannot control these values.
            use_bert=INSTRUCTOR_METRIC_SETTINGS["use_bert"],
            bert_language=INSTRUCTOR_METRIC_SETTINGS["bert_language"],
            comet_scorer=None,
        )

        # ----------------------------------------------------
        # Build base submission payload
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Add research metrics payload
        # ----------------------------------------------------

        submission.update(
            build_research_metrics_payload(
                results,
                research_mode=INSTRUCTOR_METRIC_SETTINGS["research_mode"],
            )
        )

        # ----------------------------------------------------
        # Save to Supabase
        # ----------------------------------------------------

        save_submission(submission)

    st.success("Submission saved successfully to Supabase.")

    # --------------------------------------------------------
    # Show feedback and metrics to students after saving
    # --------------------------------------------------------

    display_student_feedback_and_metrics(
        results,
        INSTRUCTOR_METRIC_SETTINGS,
    )
