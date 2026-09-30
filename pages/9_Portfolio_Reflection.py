"""Structured reflections for the three TRS430 portfolio workflows."""

import streamlit as st
from supabase import create_client

from modules.task_mode import ADAPTIVE_TRANSLATION, POST_EDITING, TRANSLATION


st.title("Portfolio Reflection")
st.write(
    "Complete one reflection after each portfolio task. Your response records "
    "your decision-making in the no-AI, AI-assisted, and EduApp post-editing workflows."
)


@st.cache_resource
def get_supabase_client():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        st.error("Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to Streamlit Secrets.")
        st.stop()


def word_count(*responses):
    return len(" ".join(part.strip() for part in responses if part).split())


def prompt_set(task_type):
    if task_type == TRANSLATION:
        return (
            "Independent decision",
            "Explain one important Arabic-to-English translation decision. Refer to the source wording and your final English solution.",
            "Diplomatic register",
            "Explain how you handled a diplomatic, institutional, or culturally significant expression.",
            "Challenge and strategy",
            "Identify a difficulty and explain the resources or reasoning you used without AI.",
            "Learning",
            "What would you do differently in a future independent translation?",
        )
    if task_type == ADAPTIVE_TRANSLATION:
        return (
            "AI contribution",
            "Explain one useful and one problematic AI suggestion. State what you retained, changed, or rejected.",
            "Prompt and verification",
            "Describe how you prompted the AI and how you checked its output against the Arabic source.",
            "Diplomatic register",
            "Explain how you ensured that modality, stance, and institutional terminology were appropriate in English.",
            "Learning",
            "What did AI add to your workflow, and what responsibility remained yours?",
        )
    return (
        "Substantive edit",
        "Describe a machine-translation error that required a substantial intervention. Quote or summarise the problem and your solution.",
        "Editing priorities",
        "Which error categories required the most work: accuracy, terminology, grammar, cohesion, register, or style? Explain why.",
        "EduApp process",
        "Use your EduApp activity to explain how you approached post-editing and revision.",
        "Learning",
        "What does this task show about the difference between post-editing and translating from scratch?",
    )


supabase = get_supabase_client()

st.header("Task record")
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("Student ID")
    student_name = st.text_input("Student name")
    assignment_code = st.selectbox(
        "Portfolio task",
        ["T1 No AI Translation", "T2 AI Assisted Translation", "T3 EduApp Post Editing"],
    )
with col2:
    group_name = st.text_input("Group")
    semester = st.text_input("Semester", value="Fall 2026")
    source_url = st.text_input("Source text URL")

task_type = {
    "T1 No AI Translation": TRANSLATION,
    "T2 AI Assisted Translation": ADAPTIVE_TRANSLATION,
    "T3 EduApp Post Editing": POST_EDITING,
}[assignment_code]

ai_tool = ""
if task_type == ADAPTIVE_TRANSLATION:
    ai_tool = st.text_input("AI system and model or version used")

st.header("Reflection")
st.caption("Write at least 200 words across the four responses. Be specific and use examples from your task.")
labels = prompt_set(task_type)
responses = []
for index in range(0, len(labels), 2):
    responses.append(
        st.text_area(
            labels[index],
            placeholder=labels[index + 1],
            height=115,
            key=f"{task_type}_{index}",
        )
    )

if st.button("Save Reflection", type="primary"):
    if not student_id.strip():
        st.error("Please enter your student ID.")
        st.stop()
    if not all(response.strip() for response in responses):
        st.error("Please complete all four reflection prompts.")
        st.stop()
    if task_type == ADAPTIVE_TRANSLATION and not ai_tool.strip():
        st.error("Please identify the AI system used.")
        st.stop()
    if word_count(*responses) < 200:
        st.error("Your combined reflection must contain at least 200 words.")
        st.stop()

    reflection_text = "\n\n".join(
        f"{labels[index]}:\n{responses[index // 2].strip()}"
        for index in range(0, len(labels), 2)
    )
    record = {
        "student_id": student_id.strip(),
        "student_name": student_name.strip() or None,
        "group_name": group_name.strip() or None,
        "semester": semester.strip() or None,
        "assignment_code": assignment_code.split()[0],
        "assignment_title": assignment_code,
        "task_type": task_type,
        "source_url": source_url.strip() or None,
        "ai_tool": ai_tool.strip() or None,
        "reflection_text": reflection_text,
    }
    try:
        supabase.table("portfolio_reflections").insert(record).execute()
    except Exception as error:
        st.error("Could not save the reflection. Run supabase/migrations/002_add_portfolio_reflections.sql once in the Supabase SQL Editor.")
        st.code(str(error))
        st.stop()

    st.success("Reflection saved successfully.")
