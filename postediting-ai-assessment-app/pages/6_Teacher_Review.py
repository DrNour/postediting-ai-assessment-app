import json

import streamlit as st

from database import create_tables, load_ai_feedback, load_teacher_reviews, save_teacher_review

create_tables()

st.set_page_config(page_title="Teacher Review", page_icon="✅", layout="wide")
st.title("Teacher Review of AI Feedback")
st.write("Approve, edit, or reject AI-generated draft feedback.")

ai_df = load_ai_feedback()

if ai_df.empty:
    st.warning("No AI feedback has been saved yet.")
else:
    st.subheader("Select AI Feedback")
    feedback_options = ai_df["feedback_id"].tolist()
    selected_feedback_id = st.selectbox("Feedback ID", feedback_options)
    selected_row = ai_df[ai_df["feedback_id"] == selected_feedback_id].iloc[0]

    st.divider()
    st.subheader("Translation Segment")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Source Text**")
        st.write(selected_row["source_text"])
    with col2:
        st.markdown("**Machine Translation**")
        st.write(selected_row["mt_output"])
    with col3:
        st.markdown("**Post-Edited Text**")
        st.write(selected_row["post_edited_text"])

    st.divider()
    st.subheader("AI Draft Feedback")
    st.markdown("**AI Overall Comment**")
    st.write(selected_row["ai_overall_comment"])

    st.markdown("**AI Risk Level**")
    risk_level = selected_row["ai_risk_level"]
    if risk_level == "low":
        st.success("Low risk")
    elif risk_level == "medium":
        st.warning("Medium risk")
    else:
        st.error("High risk")

    st.markdown("**AI Possible Errors**")
    try:
        st.json(json.loads(selected_row["ai_possible_errors"]))
    except Exception:
        st.write(selected_row["ai_possible_errors"])

    st.markdown("**AI Rubric Scores**")
    try:
        st.json(json.loads(selected_row["ai_rubric_scores"]))
    except Exception:
        st.write(selected_row["ai_rubric_scores"])

    st.divider()
    st.subheader("Teacher Decision")

    review_status = st.selectbox(
        "Review Status",
        ["approved", "edited", "rejected", "needs_discussion"],
    )
    teacher_final_feedback = st.text_area(
        "Teacher Final Feedback",
        value=selected_row["ai_overall_comment"] or "",
        height=180,
    )
    teacher_notes = st.text_area(
        "Teacher Notes",
        placeholder="Explain why you approved, edited, or rejected the AI feedback.",
        height=120,
    )
    usefulness_rating = st.slider(
        "How useful was the AI feedback?",
        min_value=1,
        max_value=5,
        value=3,
    )
    teacher_review_time_seconds = st.number_input(
        "Teacher Review Time in Seconds",
        min_value=0.0,
        value=0.0,
    )
    reviewer_id = st.text_input("Reviewer ID", value="Teacher_1")
    review_id = st.text_input("Review ID", value=f"RV_{selected_feedback_id}")

    if st.button("Save Teacher Review"):
        save_teacher_review(
            review_id=review_id,
            feedback_id=selected_feedback_id,
            segment_id=selected_row["segment_id"],
            review_status=review_status,
            teacher_final_feedback=teacher_final_feedback,
            teacher_notes=teacher_notes,
            reviewer_id=reviewer_id,
            usefulness_rating=usefulness_rating,
            teacher_review_time_seconds=teacher_review_time_seconds,
        )
        st.success("Teacher review saved successfully.")

st.divider()
st.subheader("Saved Teacher Reviews")
reviews_df = load_teacher_reviews()
if reviews_df.empty:
    st.info("No teacher reviews saved yet.")
else:
    st.dataframe(reviews_df, use_container_width=True)
