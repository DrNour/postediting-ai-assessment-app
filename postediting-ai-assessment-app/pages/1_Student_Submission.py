import streamlit as st
from database import add_segment, add_student, add_task, create_tables

create_tables()

st.set_page_config(page_title="Student Submission", page_icon="✍️", layout="wide")
st.title("Student Submission")
st.write("Save a student's post-edited translation segment.")

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
    task_id = st.text_input("Task ID", value="T001")
    task_title = st.text_input("Task Title", value="University Announcement")
    domain = st.text_input("Domain", value="Institutional")
with col4:
    source_language = st.text_input("Source Language", value="Arabic")
    target_language = st.text_input("Target Language", value="English")

st.header("Translation Segment")
segment_id = st.text_input("Segment ID", value="SEG001")
source_text = st.text_area("Source Text", value="زار الوزير الجامعة أمس.", height=120)
mt_output = st.text_area("Machine Translation", value="The minister visited the university yesterday.", height=120)
post_edited_text = st.text_area("Post-Edited Text", value="The minister visited the university yesterday.", height=120)
editing_time_seconds = st.number_input("Editing Time in Seconds", min_value=0.0, value=52.0)

if st.button("Save Submission"):
    add_student(student_id, student_name, group_name, semester)
    add_task(task_id, task_title, source_language, target_language, domain)
    add_segment(segment_id, student_id, task_id, source_text, mt_output, post_edited_text, editing_time_seconds)
    st.success("Submission saved successfully.")
