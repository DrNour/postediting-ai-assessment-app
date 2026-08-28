"""Streamlit entrypoint and navigation router for EduApp."""

import streamlit as st


st.set_page_config(
    page_title="EduApp | Translation and Post-Editing",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared styling. Keep it theme-safe: do not force all text to a fixed colour.
st.markdown(
    """
    <style>
    .track-box {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 12px;
        padding: 18px;
        line-height: 1.9;
        font-size: 1rem;
    }
    .deleted-word {
        color: #991b1b;
        background-color: #fee2e2;
        text-decoration: line-through;
        padding: 2px 4px;
        border-radius: 4px;
        margin: 1px;
    }
    .added-word {
        color: #065f46;
        background-color: #d1fae5;
        font-weight: 700;
        padding: 2px 4px;
        border-radius: 4px;
        margin: 1px;
    }
    .same-word {
        padding: 2px 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = {
    "Start": [
        st.Page(
            "views/Home.py",
            title="Home / Workflow",
            icon="🏠",
            default=True,
        ),
    ],
    "Student": [
        st.Page(
            "views/Student_Assignments.py",
            title="Student Assignments",
            icon="✍️",
        ),
        st.Page(
            "pages/1_Student_Submission.py",
            title="Manual / Pilot Submission",
            icon="🧪",
        ),
    ],
    "Teacher": [
        st.Page(
            "views/Teacher_Assignments.py",
            title="Create Assignments",
            icon="📁",
        ),
        st.Page(
            "views/Teacher_Submissions.py",
            title="Submissions and Scoring",
            icon="📥",
        ),
        st.Page(
            "pages/2_Teacher_Annotation.py",
            title="Teacher Annotation",
            icon="📝",
        ),
        st.Page(
            "pages/8_Teacher_Review.py",
            title="AI Feedback Review",
            icon="✅",
        ),
    ],
    "AI and Research": [
        st.Page("pages/3_Dashboard.py", title="Dashboard", icon="📊"),
        st.Page(
            "pages/4_Error_Classifier.py",
            title="Error Classifier",
            icon="🏷️",
        ),
        st.Page("pages/5_AI_Feedback.py", title="AI Feedback", icon="💬"),
        st.Page(
            "pages/6_Research_Analytics.py",
            title="Research Analytics",
            icon="🔬",
        ),
        st.Page("pages/7_Evaluation.py", title="Evaluation", icon="📈"),
    ],
}

current_page = st.navigation(pages, position="sidebar")
current_page.run()
