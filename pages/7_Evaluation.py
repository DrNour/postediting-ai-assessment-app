import pandas as pd
import streamlit as st

from database import add_research_columns, create_tables, load_evaluation_dataset
from modules.evaluation import (
    calculate_ai_teacher_agreement,
    calculate_review_status_counts,
    prepare_evaluation_df,
    summarise_usefulness,
    usefulness_by_review_status,
)

create_tables()
add_research_columns()

st.set_page_config(page_title="Research Evaluation", page_icon="📈", layout="wide")
st.title("Research Evaluation and Export")
st.write("Prepare dissertation-ready evaluation data from teacher annotations, AI feedback, and review decisions.")

df = load_evaluation_dataset()

if df.empty:
    st.warning("No evaluation data available yet.")
else:
    eval_df = prepare_evaluation_df(df)

    st.subheader("Evaluation Dataset")
    st.dataframe(eval_df, use_container_width=True)

    st.download_button(
        label="Download Evaluation Dataset as CSV",
        data=eval_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="postediting_evaluation_dataset.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Teacher Review Outcomes")
    review_counts = calculate_review_status_counts(eval_df)
    if review_counts.empty:
        st.info("No teacher review outcomes available yet.")
    else:
        st.dataframe(review_counts, use_container_width=True)
        st.bar_chart(review_counts.set_index("review_status")["count"])

    st.divider()
    st.subheader("AI vs Teacher Agreement")
    agreement = calculate_ai_teacher_agreement(eval_df)
    if "error" in agreement:
        st.warning(agreement["error"])
    else:
        st.metric("Cohen's Kappa", round(agreement["kappa"], 3))
        report_df = pd.DataFrame(agreement["classification_report"]).transpose()
        st.markdown("**Classification Report**")
        st.dataframe(report_df, use_container_width=True)
        st.markdown("**Compared Cases**")
        st.dataframe(
            agreement["comparison_df"][
                [
                    "segment_id",
                    "teacher_category",
                    "ai_category",
                    "ai_confidence",
                    "review_status",
                    "ai_risk_level",
                ]
            ],
            use_container_width=True,
        )

    st.divider()
    st.subheader("Teacher Usefulness Ratings")
    usefulness = summarise_usefulness(eval_df)
    if "error" in usefulness:
        st.info(usefulness["error"])
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Usefulness", usefulness["mean"])
        with col2:
            st.metric("Median", usefulness["median"])
        with col3:
            st.metric("Number of Ratings", usefulness["count"])
        st.write(usefulness)

    st.divider()
    st.subheader("Usefulness by Review Status")
    usefulness_status = usefulness_by_review_status(eval_df)
    if usefulness_status.empty:
        st.info("No usefulness-by-status summary available yet.")
    else:
        st.dataframe(usefulness_status, use_container_width=True)

    st.divider()
    st.subheader("Dissertation-Ready Tables")
    st.markdown(
        """
Use the tables above in your results chapter:

1. Teacher review outcomes
2. AI vs teacher agreement
3. Classification report
4. Usefulness ratings
5. Usefulness by review status
"""
    )
