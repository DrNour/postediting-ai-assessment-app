import json

import streamlit as st

from database import create_tables, get_segment_by_id, get_segment_ids, save_ai_feedback
from modules.ai_feedback import check_feedback_risk, generate_ai_feedback
from modules.metrics import compare_mt_pe

try:
    from modules.similarity import semantic_similarity
except Exception:
    semantic_similarity = None

create_tables()

st.set_page_config(page_title="AI Feedback", page_icon="💬", layout="wide")
st.title("AI Feedback Generator")
st.write("Generate draft AI feedback for a post-edited translation. Teacher review is required.")
st.warning("AI-generated feedback is a draft and must be reviewed by a teacher before assessment use.")

segment_ids = get_segment_ids()

if len(segment_ids) == 0:
    st.warning("No segments found. Please add student submissions first.")
else:
    selected_segment_id = st.selectbox("Select Segment", segment_ids)
    segment = get_segment_by_id(selected_segment_id)

    if segment:
        st.subheader("Segment Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Source Text**")
            st.write(segment["source_text"])
        with col2:
            st.markdown("**Machine Translation**")
            st.write(segment["mt_output"])
        with col3:
            st.markdown("**Post-Edited Text**")
            st.write(segment["post_edited_text"])

        st.divider()
        st.subheader("Automatic Metrics")
        metrics = compare_mt_pe(segment["mt_output"], segment["post_edited_text"])
        st.json(metrics)

        mt_pe_semantic = None
        source_pe_semantic = None

        if semantic_similarity is not None:
            with st.spinner("Calculating semantic similarity..."):
                mt_pe_semantic = semantic_similarity(segment["mt_output"], segment["post_edited_text"])
                source_pe_semantic = semantic_similarity(segment["source_text"], segment["post_edited_text"])
            col4, col5 = st.columns(2)
            with col4:
                st.metric("MT vs PE Semantic Similarity", mt_pe_semantic)
            with col5:
                st.metric("Source vs PE Semantic Similarity", source_pe_semantic)
        else:
            st.info("Semantic similarity module not available.")

        st.divider()
        model_name = st.text_input("Model name", value="gpt-4.1-mini")

        if st.button("Generate AI Feedback"):
            with st.spinner("Generating AI feedback..."):
                result = generate_ai_feedback(
                    source_text=segment["source_text"],
                    mt_output=segment["mt_output"],
                    post_edited_text=segment["post_edited_text"],
                    editing_time_seconds=segment["editing_time_seconds"],
                    lexical_similarity=metrics["lexical_similarity"],
                    change_ratio=metrics["change_ratio"],
                    mt_pe_semantic_similarity=mt_pe_semantic,
                    source_pe_semantic_similarity=source_pe_semantic,
                    model_name=model_name,
                )

            if result["success"]:
                feedback = result["feedback"]
                st.session_state["latest_feedback"] = feedback
                st.session_state["latest_feedback_segment_id"] = selected_segment_id
                st.success("AI feedback generated.")

                st.subheader("Overall Comment")
                st.write(feedback.get("overall_comment", ""))
                st.subheader("Possible Errors")
                st.json(feedback.get("possible_errors", []))
                st.subheader("Rubric Scores")
                st.json(feedback.get("rubric_scores", {}))

                risk = check_feedback_risk(feedback)
                st.session_state["latest_feedback_risk"] = risk
                st.subheader("Risk / Hallucination Warning")
                st.write(f"Risk level: **{risk['risk_level']}**")
                if risk["warnings"]:
                    for warning in risk["warnings"]:
                        st.warning(warning)
                else:
                    st.success("No major warning detected.")

                st.subheader("Raw JSON")
                st.code(json.dumps(feedback, ensure_ascii=False, indent=2), language="json")
            else:
                st.error(result["error"])
                st.subheader("Raw Model Output")
                st.code(result["raw_output"])

        if "latest_feedback" in st.session_state and st.session_state.get("latest_feedback_segment_id") == selected_segment_id:
            feedback = st.session_state["latest_feedback"]
            risk = st.session_state.get("latest_feedback_risk", check_feedback_risk(feedback))
            feedback_id = st.text_input("Feedback ID", value=f"FB_{selected_segment_id}")
            if st.button("Save AI Feedback for Teacher Review"):
                save_ai_feedback(
                    feedback_id=feedback_id,
                    segment_id=selected_segment_id,
                    ai_overall_comment=feedback.get("overall_comment", ""),
                    ai_possible_errors=feedback.get("possible_errors", []),
                    ai_rubric_scores=feedback.get("rubric_scores", {}),
                    ai_raw_json=feedback,
                    ai_risk_level=risk["risk_level"],
                )
                st.success("AI feedback saved for teacher review.")
