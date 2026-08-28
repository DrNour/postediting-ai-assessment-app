"""Manual/pilot submission page supporting translation and post-editing."""

import pandas as pd
import streamlit as st
from supabase import create_client

from metrics import build_research_metrics_payload, compare_postedit_with_raw_mt
from modules.task_mode import (
    POST_EDITING,
    TASK_OPTIONS,
    is_translation,
    make_translation_metrics_task_appropriate,
    normalize_task_type,
    student_output_label,
    task_instruction,
    task_type_label,
)


st.title("Manual / Pilot Submission")
st.write(
    "Enter a standalone translation or post-editing record. For normal class use, "
    "the assignment-based **Student Assignments** page is recommended."
)


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


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def clean_value_for_supabase(value):
    try:
        if value is None or pd.isna(value):
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
        # The page remains usable with safe defaults when the optional settings
        # table has not yet been created.
        pass
    return settings


def save_submission(submission):
    clean_submission = {
        key: clean_value_for_supabase(value) for key, value in submission.items()
    }
    try:
        return supabase.table("submissions").insert(clean_submission).execute()
    except Exception as error:
        error_text = str(error)
        st.error("Could not save the submission to Supabase.")
        if "task_type" in error_text.lower():
            st.warning(
                "Run supabase/migrations/001_add_task_type.sql once in the "
                "Supabase SQL Editor, then submit again."
            )
        st.write("Submission data being sent:")
        st.json(clean_submission)
        st.write("Supabase error:")
        st.code(error_text)
        st.stop()


def display_student_feedback_and_metrics(results, settings, task_type):
    if not settings.get("show_student_metrics", True):
        return

    st.header("Feedback and Metrics")

    if not is_translation(task_type) and settings.get("show_editing_summary", True):
        st.subheader("Post-editing effort")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Raw MT words", format_metric(results.get("raw_mt_word_count"), 0))
        with col2:
            st.metric("Post-edited words", format_metric(results.get("pe_word_count"), 0))
        with col3:
            st.metric(
                "Word-count difference",
                format_metric(results.get("mt_pe_word_count_difference"), 0),
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("MT–PE similarity", format_metric(results.get("mt_pe_cosine_similarity"), 3))
        with col5:
            st.metric("Edit-distance ratio", format_metric(results.get("mt_pe_edit_distance_ratio"), 3))
        with col6:
            st.metric("Length ratio", format_metric(results.get("mt_pe_length_ratio"), 3))

        col7, col8, col9 = st.columns(3)
        with col7:
            st.metric("Inserted words", format_metric(results.get("mt_pe_inserted_words"), 0))
        with col8:
            st.metric("Deleted words", format_metric(results.get("mt_pe_deleted_words"), 0))
        with col9:
            st.metric("Replaced words", format_metric(results.get("mt_pe_replaced_words"), 0))

    if (
        not is_translation(task_type)
        and settings.get("show_mt_pe_overlap_metrics", True)
    ):
        st.subheader("MT–post-edit overlap")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BLEU against raw MT", format_metric(results.get("mt_pe_overlap_bleu"), 2))
        with col2:
            st.metric("chrF against raw MT", format_metric(results.get("mt_pe_overlap_chrf"), 2))
        with col3:
            st.metric("TER against raw MT", format_metric(results.get("mt_pe_overlap_ter"), 2))
        st.caption(
            "These metrics describe editing overlap and effort, not final translation quality."
        )

    if settings.get("show_reference_quality_metrics", True):
        st.subheader("Reference-based quality")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Output BLEU", format_metric(results.get("pe_quality_bleu"), 2))
        with col2:
            st.metric("Output chrF", format_metric(results.get("pe_quality_chrf"), 2))
        with col3:
            st.metric("Output TER", format_metric(results.get("pe_quality_ter"), 2))
        st.caption(
            "These scores are quality-oriented only when an independent reference "
            "translation is available."
        )

    if settings.get("use_bert", False) and settings.get("run_advanced_metrics_now", False):
        st.subheader("Advanced metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Student-output BERTScore F1",
                format_metric(results.get("pe_quality_bertscore_f1"), 4),
            )
        with col2:
            st.metric(
                "Human-translation BERTScore F1",
                format_metric(results.get("ht_quality_bertscore_f1"), 4),
            )

    if settings.get("show_automated_interpretation", True):
        interpretation = results.get("mt_pe_interpretation")
        if interpretation:
            st.subheader("Automated interpretation")
            st.info(interpretation)


metric_settings = load_metric_settings()

st.header("Student Information")
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("Student ID", value="S001")
    student_name = st.text_input("Student Name", value="Student A")
with col2:
    group_name = st.text_input("Group", value="Group 1")
    semester = st.text_input("Semester", value="Fall 2026")

st.header("Task Information")
task_label = st.radio(
    "Task type",
    list(TASK_OPTIONS),
    horizontal=True,
)
task_type = TASK_OPTIONS[task_label]
st.info(task_instruction(task_type))

col3, col4 = st.columns(2)
with col3:
    assignment_id = st.text_input("Assignment ID", value="T001")
    assignment_title = st.text_input("Assignment Title", value="University Announcement")
    domain = st.text_input("Domain", value="Institutional")
with col4:
    source_language = st.text_input("Source Language", value="Arabic")
    target_language = st.text_input("Target Language", value="English")

st.header("Texts")
source_text = st.text_area(
    "Source Text",
    value="زار الوزير الجامعة أمس.",
    height=120,
)

if is_translation(task_type):
    raw_mt = ""
    with st.expander("Optional teacher/research comparison text", expanded=False):
        raw_mt = st.text_area(
            "Raw machine translation (not part of the student's translation task)",
            value="",
            height=120,
        )
else:
    raw_mt = st.text_area(
        "Machine Translation",
        value="The minister visited the university yesterday.",
        height=120,
    )

reference_translation = st.text_area(
    "Reference Translation (optional)",
    value="The minister visited the university yesterday.",
    height=120,
)

output_label = student_output_label(task_type)
student_output = st.text_area(
    output_label,
    value=(
        "The minister visited the university yesterday."
        if not is_translation(task_type)
        else ""
    ),
    height=160,
)

student_reflection = st.text_area(
    "Reflection / Comment (optional)",
    placeholder="Briefly explain a key decision or difficulty.",
    height=100,
)

editing_time_seconds = st.number_input(
    "Task Time in Seconds",
    min_value=0.0,
    value=52.0,
)

with st.expander("Metric information", expanded=False):
    st.write(
        "Metrics are calculated automatically after submission. Post-editing effort "
        "metrics are omitted for independent translation tasks."
    )

if st.button("Save Submission", type="primary"):
    if not safe_text(student_id):
        st.error("Please enter a student ID.")
        st.stop()

    if not safe_text(student_output):
        st.error(f"Please enter the {output_label.lower()}.")
        st.stop()

    if not is_translation(task_type) and not safe_text(raw_mt):
        st.error("Please enter the machine translation for a post-editing task.")
        st.stop()

    with st.spinner("Calculating metrics and saving submission..."):
        use_bert_now = (
            metric_settings.get("research_mode", True)
            and metric_settings.get("run_advanced_metrics_now", False)
            and metric_settings.get("use_bert", False)
        )

        results = compare_postedit_with_raw_mt(
            raw_mt=raw_mt,
            post_edited_text=student_output,
            human_translation=student_output if is_translation(task_type) else None,
            reference_text=reference_translation,
            source_text=source_text,
            teacher_score=None,
            teacher_feedback="",
            use_bert=use_bert_now,
            bert_language=metric_settings.get("bert_language", "en"),
            comet_scorer=None,
        )

        if is_translation(task_type):
            make_translation_metrics_task_appropriate(
                results,
                has_reference=bool(safe_text(reference_translation)),
            )

        submission = {
            "assignment_code": assignment_id,
            "task_id": assignment_id,
            "assignment_title": assignment_title,
            "task_type": normalize_task_type(task_type),
            "student_id": student_id,
            "student_name": student_name,
            "source_text": source_text,
            "machine_translation": raw_mt,
            "reference_translation": reference_translation,
            # Legacy compatibility: this remains the canonical final-output column
            # for both task modes.
            "post_edited_text": student_output,
            "student_reflection": student_reflection,
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
                research_mode=metric_settings.get("research_mode", True),
            )
        )

        save_submission(submission)

    st.success(
        f"{task_type_label(task_type)} submission saved successfully to Supabase."
    )
    display_student_feedback_and_metrics(results, metric_settings, task_type)
