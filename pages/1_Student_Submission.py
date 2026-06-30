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
st.write("Students submit post-edited translations. Metrics are controlled by the instructor.")


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
# Default metric settings
# Used if Supabase settings table is missing or unavailable.
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


def clean_value_for_supabase(value):
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return value

    except Exception:
        return value


def format_metric(value, digits=3):
    if value is None:
        return "N/A"

    try:
        return round(float(value), digits)
    except Exception:
        return value


def load_metric_settings():
    """
    Loads instructor-controlled metric settings from Supabase.
    Students cannot change these settings.
    """

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
        # Do not break student submission if the settings table is unavailable.
        pass

    return settings


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


def display_student_feedback_and_metrics(results, settings):
    """
    Shows students the instructor-approved metrics.
    Students can view results but cannot control which metrics are calculated.
    """

    if not settings.get("show_student_metrics", True):
        return

    st.header("Feedback and Metrics")

    # --------------------------------------------------------
    # Post-editing effort
    # --------------------------------------------------------

    if settings.get("show_editing_summary", True):
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
    # MT-post-edit overlap
    # --------------------------------------------------------

    if settings.get("show_mt_pe_overlap_metrics", True):
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
    # Reference-based quality
    # --------------------------------------------------------

    if settings.get("show_reference_quality_metrics", True):
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
            "These metrics are quality-oriented only when an independent reference translation is available."
        )

    # --------------------------------------------------------
    # Advanced metrics
    # --------------------------------------------------------

    if settings.get("use_bert", False) and settings.get("run_advanced_metrics_now", False):
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
    # Automated interpretation
    # --------------------------------------------------------

    if settings.get("show_automated_interpretation", True):
        interpretation = results.get("mt_pe_interpretation")

        if interpretation:
            st.subheader("Automated interpretation")
            st.info(interpretation)


# ============================================================
# Load instructor metric settings
# ============================================================

metric_settings = load_metric_settings()


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

student_reflection = st.text_area(
    "Reflection / Comment (optional)",
    placeholder=(
        "Briefly explain the main change you made and why, "
        "or describe any difficulty you noticed."
    ),
    height=100,
)

editing_time_seconds = st.number_input(
    "Editing Time in Seconds",
    min_value=0.0,
    value=52.0,
)


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

        use_bert_now = (
            metric_settings.get("research_mode", True)
            and metric_settings.get("run_advanced_metrics_now", False)
            and metric_settings.get("use_bert", False)
        )

        results = compare_postedit_with_raw_mt(
            raw_mt=raw_mt,
            post_edited_text=post_edited_text,
            human_translation=None,
            reference_text=reference_translation,
            source_text=source_text,
            teacher_score=None,
            teacher_feedback="",
            use_bert=use_bert_now,
            bert_language=metric_settings.get("bert_language", "en"),
            comet_scorer=None,
        )
        submission = {
            "assignment_code": assignment_id,
            "task_id": assignment_id,
            "assignment_title": assignment_title,
            "student_id": student_id,
            "student_name": student_name,
            "source_text": source_text,
            "machine_translation": raw_mt,
            "reference_translation": reference_translation,
            "post_edited_text": post_edited_text,
            "student_reflection": student_reflection,
            "editing_time_seconds": editing_time_seconds,
            "group_name": group_name,
            "semester": semester,
            "domain": domain,
            "source_language": source_language,
            "target_language": target_language,
            "teacher_score": None,
            "teacher_feedback": "", }

        submission.update(
            build_research_metrics_payload(
                results,
                research_mode=metric_settings.get("research_mode", True),
            )
        )

        save_submission(submission)

    st.success("Submission saved successfully to Supabase.")

    display_student_feedback_and_metrics(
        results,
        metric_settings,
    )
