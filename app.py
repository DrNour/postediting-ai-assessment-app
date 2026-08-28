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

    /* Keep interface text crisp instead of Streamlit's muted/faded treatment. */
    [data-testid="stSidebarNav"] *,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] {
        opacity: 1 !important;
        color: var(--text-color) !important;
    }

    [data-testid="stSidebarNav"] span,
    [data-testid="stSidebarNav"] p,
    [data-testid="stSidebarNav"] a {
        font-weight: 600 !important;
    }

    /* Make secondary/help text readable while still adapting to light/dark themes. */
    [data-testid="stCaptionContainer"] p,
    [data-testid="InputInstructions"] {
        opacity: 1 !important;
        color: var(--text-color) !important;
    }

    /* Streamlit/browser styles fade disabled text areas by default.
       Keep read-only source/MT text fully legible. */
    textarea:disabled,
    input:disabled,
    [data-baseweb="textarea"] textarea:disabled,
    [data-baseweb="input"] input:disabled {
        opacity: 1 !important;
        color: var(--text-color) !important;
        -webkit-text-fill-color: var(--text-color) !important;
    }

    [data-baseweb="textarea"] textarea:disabled::placeholder,
    [data-baseweb="input"] input:disabled::placeholder {
        opacity: 1 !important;
        color: var(--text-color) !important;
        -webkit-text-fill-color: var(--text-color) !important;
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
