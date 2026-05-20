import streamlit as st
from database import create_tables, load_segments_with_annotations, load_teacher_reviews
from modules.metrics import compare_mt_pe

create_tables()

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("Post-Editing Dashboard")

df = load_segments_with_annotations()

if df.empty:
    st.warning("No data available yet.")
else:
    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Segments", len(df))
    with col2:
        avg_time = df["editing_time_seconds"].mean()
        st.metric("Average Editing Time", round(avg_time, 2))
    with col3:
        annotated_count = df["category"].notna().sum()
        st.metric("Annotated Segments", annotated_count)

    st.subheader("MT vs Post-Edited Metrics")
    metric_rows = []
    for _, row in df.iterrows():
        metrics = compare_mt_pe(row["mt_output"], row["post_edited_text"])
        metrics["segment_id"] = row["segment_id"]
        metric_rows.append(metrics)
    if metric_rows:
        import pandas as pd
        metrics_df = pd.DataFrame(metric_rows)
        st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Error Category Counts")
    if df["category"].notna().sum() > 0:
        st.bar_chart(df["category"].value_counts())
    else:
        st.info("No annotations yet.")

    st.subheader("Severity Counts")
    if df["severity"].notna().sum() > 0:
        st.bar_chart(df["severity"].value_counts())
    else:
        st.info("No severity labels yet.")

st.divider()
st.subheader("Teacher Review Analytics")
reviews_df = load_teacher_reviews()
if reviews_df.empty:
    st.info("No teacher review data yet.")
else:
    st.dataframe(reviews_df, use_container_width=True)
    review_counts = reviews_df["review_status"].value_counts()
    st.bar_chart(review_counts)
    approved_count = (reviews_df["review_status"] == "approved").sum()
    edited_count = (reviews_df["review_status"] == "edited").sum()
    rejected_count = (reviews_df["review_status"] == "rejected").sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Approved", approved_count)
    with col2:
        st.metric("Edited", edited_count)
    with col3:
        st.metric("Rejected", rejected_count)
