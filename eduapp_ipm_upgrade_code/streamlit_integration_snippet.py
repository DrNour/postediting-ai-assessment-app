"""Minimal Streamlit integration for EduApp.

Add this after the user submits a post-editing task. The snippet computes lightweight
metrics immediately. Run COMET/BERTScore/LLM judge offline in batch to avoid making
students wait during submission.
"""

import streamlit as st

from eduapp_metrics.metrics import compute_metrics_for_record
from eduapp_metrics.schemas import MetricConfig, SegmentRecord


def compute_submission_metrics(
    source_text: str,
    mt_output: str,
    user_submission: str,
    time_spent_sec: float,
    participant_id: str,
    exercise_id: str,
    instance_id: str,
):
    record = SegmentRecord(
        instance_id=instance_id,
        participant_id=participant_id,
        exercise_id=exercise_id,
        task_type="post_editing" if mt_output else "translation",
        source_text=source_text,
        mt_output=mt_output,
        user_submission=user_submission,
        time_spent_sec=time_spent_sec,
    )
    config = MetricConfig(
        use_sacrebleu=True,
        use_bertscore=False,
        use_comet=False,
        use_qe=False,
        use_embeddings=False,
        use_llm_judge=False,
    )
    result = compute_metrics_for_record(record, config)
    return result.metrics


# Example inside your submit button logic:
# if st.button("Submit"):
#     metrics = compute_submission_metrics(
#         source_text=source_text,
#         mt_output=mt_output,
#         user_submission=st.session_state.user_submission,
#         time_spent_sec=time.time() - st.session_state.start_time,
#         participant_id=participant_id,
#         exercise_id=exercise_id,
#         instance_id=instance_id,
#     )
#     st.session_state.report_payload.update(metrics)
#     st.success("Submission saved with effort metrics.")
