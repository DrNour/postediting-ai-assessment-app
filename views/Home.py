"""Public home page. It remains visible even before Supabase is configured."""

import streamlit as st

from modules.config import get_secret, missing_secret_names


st.title("EduApp: Translation and Post-Editing Assessment")

st.write(
    """
EduApp supports two student pathways: **translate from the source text** or
**post-edit a machine translation**. Teachers can create assignments, review
submissions, annotate errors, generate draft AI feedback, and export research data.
"""
)

st.info("AI suggests. The teacher decides.")

st.header("Workflow")

col1, col2 = st.columns(2)

with col1:
    st.subheader("For students")
    st.markdown(
        """
1. Open **Student Assignments**.
2. Choose an active assignment.
3. Select **Translation** or **Post-editing**.
4. Complete and submit the task.
"""
    )

with col2:
    st.subheader("For teachers")
    st.markdown(
        """
1. Create assignments and add source, optional MT, and reference texts.
2. Review and score student submissions.
3. Annotate errors or generate draft AI feedback.
4. Export data for analysis.
"""
    )

st.header("Setup status")

required = ["SUPABASE_URL", "SUPABASE_KEY", "TEACHER_PASSWORD"]
missing = missing_secret_names(required)

if missing:
    st.warning(
        "The interface is available, but data pages need Streamlit Secrets. "
        f"Missing: {', '.join(missing)}. Copy `.streamlit/secrets.toml.example` "
        "to `.streamlit/secrets.toml`, add the values, and run the Supabase SQL setup."
    )
else:
    st.success("Supabase and teacher-access secrets are configured.")

if not get_secret("OPENAI_API_KEY"):
    st.caption("OPENAI_API_KEY is optional and is needed only for AI feedback.")

st.warning(
    "Use anonymised student identifiers and appropriate access controls for real classroom data."
)
