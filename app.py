
import streamlit as st
from database import create_tables, add_research_columns
from pathlib import Path
from datetime import datetime
import uuid
import pandas as pd
# ============================================================
# Assignment system
# ============================================================

ASSIGNMENTS_FILE = Path("data/assignments.csv")

ASSIGNMENT_COLUMNS = [
    "assignment_id",
    "created_at",
    "course",
    "title",
    "instructions",
    "source_text",
    "machine_translation",
    "due_date",
    "max_score",
    "active",
]


def ensure_assignment_storage():
    """Create the data folder and assignments file if they do not exist."""
    Path("data").mkdir(exist_ok=True)

    if not ASSIGNMENTS_FILE.exists():
        df = pd.DataFrame(columns=ASSIGNMENT_COLUMNS)
        df.to_csv(ASSIGNMENTS_FILE, index=False)


def load_assignments():
    """Load assignments from CSV."""
    ensure_assignment_storage()
    return pd.read_csv(ASSIGNMENTS_FILE)


def save_assignment(assignment):
    """Save a new assignment to CSV."""
    ensure_assignment_storage()

    df = load_assignments()
    new_row = pd.DataFrame([assignment])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ASSIGNMENTS_FILE, index=False)


def get_teacher_password():
    """
    Reads the teacher password from Streamlit secrets.
    If no secret exists, it uses a temporary default password.
    Change this later before real deployment.
    """
    try:
        return st.secrets.get("TEACHER_PASSWORD", "teacher123")
    except Exception:
        return "teacher123"


def teacher_login():
    """Simple teacher password protection."""
    st.subheader("Teacher Login")

    password = st.text_input("Enter teacher password", type="password")

    if password == get_teacher_password():
        st.success("Teacher access granted.")
        return True

    if password:
        st.error("Incorrect password.")

    return False


def teacher_assignment_page():
    """Page where teachers create and view assignments."""
    st.title("Teacher Assignment Creator")

    if not teacher_login():
        st.info("Enter the teacher password to create assignments.")
        return

    st.divider()

    st.subheader("Create a New Assignment")

    with st.form("create_assignment_form"):
        course = st.text_input("Course name", placeholder="Example: Translation Studies")
        title = st.text_input("Assignment title", placeholder="Example: Post-editing Task 1")

        instructions = st.text_area(
            "Instructions for students",
            placeholder="Explain what students should do.",
            height=120,
        )

        source_text = st.text_area(
            "Source text",
            placeholder="Paste the original text here.",
            height=180,
        )

        machine_translation = st.text_area(
            "Machine translation",
            placeholder="Paste the machine-translated text here.",
            height=180,
        )

        due_date = st.date_input("Due date")

        max_score = st.number_input(
            "Maximum score",
            min_value=1,
            max_value=100,
            value=10,
        )

        active = st.checkbox("Make this assignment visible to students", value=True)

        submitted = st.form_submit_button("Create Assignment")

        if submitted:
            if not title.strip():
                st.error("Please enter an assignment title.")
            elif not source_text.strip():
                st.error("Please enter the source text.")
            elif not machine_translation.strip():
                st.error("Please enter the machine translation.")
            else:
                assignment = {
                    "assignment_id": str(uuid.uuid4())[:8],
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "course": course.strip(),
                    "title": title.strip(),
                    "instructions": instructions.strip(),
                    "source_text": source_text.strip(),
                    "machine_translation": machine_translation.strip(),
                    "due_date": str(due_date),
                    "max_score": max_score,
                    "active": "yes" if active else "no",
                }

                save_assignment(assignment)

                st.success("Assignment created successfully.")

    st.divider()

    st.subheader("Existing Assignments")

    assignments = load_assignments()

    if assignments.empty:
        st.info("No assignments created yet.")
    else:
        st.dataframe(
            assignments.sort_values("created_at", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def student_assignment_page():
    """Page where students can view available assignments."""
    st.title("Student Assignments")

    assignments = load_assignments()

    if assignments.empty:
        st.info("No assignments are available yet.")
        return

    active_assignments = assignments[
        assignments["active"].astype(str).str.lower() == "yes"
    ]

    if active_assignments.empty:
        st.info("No active assignments are currently available.")
        return

    assignment_titles = active_assignments["title"].tolist()

    selected_title = st.selectbox("Choose an assignment", assignment_titles)

    selected_assignment = active_assignments[
        active_assignments["title"] == selected_title
    ].iloc[0]

    st.subheader(selected_assignment["title"])

    if pd.notna(selected_assignment["course"]):
        st.write(f"**Course:** {selected_assignment['course']}")

    st.write(f"**Due date:** {selected_assignment['due_date']}")
    st.write(f"**Maximum score:** {selected_assignment['max_score']}")

    st.markdown("### Instructions")
    st.write(selected_assignment["instructions"])

    st.markdown("### Source Text")
    st.text_area(
        "Source text",
        selected_assignment["source_text"],
        height=180,
        disabled=True,
    )

    st.markdown("### Machine Translation")
    st.text_area(
        "Machine translation",
        selected_assignment["machine_translation"],
        height=180,
        disabled=True,
    )

    st.markdown("### Your Post-edited Version")

    student_answer = st.text_area(
        "Paste or type your post-edited translation here",
        height=220,
    )

    if st.button("Submit Assignment"):
        if not student_answer.strip():
            st.error("Please enter your post-edited version before submitting.")
        else:
            st.success("Your assignment response has been entered.")
            st.info(
                "Next step: we can add code to save student submissions and calculate scores."
            )
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
