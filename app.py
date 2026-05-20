
import streamlit as st
from database import create_tables, add_research_columns
from pathlib import Path
from datetime import datetime
import uuid
import pandas as pd
st.set_page_config(
    page_title="AI Post-Editing Assessment App",
    page_icon="📘",
    layout="wide"
)

create_tables()
add_research_columns()

st.title("AI-Assisted Post-Editing Assessment App")

st.write(
    """
This Streamlit app supports post-editing assessment, teacher annotation,
AI feedback, and research evaluation for translation training.

The core principle is: **AI suggests. Teacher decides.**
"""
)

st.warning(
    "Public/demo use should rely on synthetic or anonymised data only. "
    "Do not upload identifiable student information to a public deployment."
)

st.header("Workflow")
st.markdown(
    """
1. **Student Submission** - save source text, MT output, post-edited text, and editing time.
2. **Teacher Annotation** - label translation errors using a structured taxonomy.
3. **Dashboard** - view editing-time and annotation summaries.
4. **Error Classifier** - train a baseline model from teacher-labelled data.
5. **AI Feedback** - generate draft AI feedback using the taxonomy and rubric.
6. **Teacher Review** - approve, edit, or reject AI feedback.
7. **Evaluation** - export research datasets and calculate evaluation metrics.
"""
)

st.header("Recommended Use")
st.markdown(
    """
- Start with a small pilot dataset.
- Keep real student data private and anonymised.
- Use teacher review before feedback reaches students.
- Treat AI outputs as draft feedback, not final grades.
"""
)

st.success("Use the sidebar to open each page.")
