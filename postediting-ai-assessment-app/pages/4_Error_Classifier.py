import pandas as pd
import streamlit as st

from database import create_tables, get_segment_by_id, get_segment_ids, load_segments_with_annotations
from modules.error_classifier import (
    load_model,
    predict_error_category,
    save_model,
    train_and_evaluate_classifier,
)

create_tables()

st.set_page_config(page_title="Error Classifier", page_icon="🤖", layout="wide")
st.title("Translation Error Classification Model")
st.write("Train a baseline machine learning model using teacher-labelled error categories.")

df = load_segments_with_annotations()
labelled_df = df.dropna(subset=["category"])

st.subheader("Labelled Dataset")
st.metric("Labelled Segments", len(labelled_df))

if len(labelled_df) > 0:
    st.dataframe(
        labelled_df[
            [
                "segment_id",
                "source_text",
                "mt_output",
                "post_edited_text",
                "category",
                "subcategory",
                "severity",
            ]
        ],
        use_container_width=True,
    )

st.divider()
st.subheader("Train Model")

if st.button("Train Error Classifier"):
    result = train_and_evaluate_classifier(df)
    if "error" in result:
        st.error(result["error"])
    else:
        st.success("Model trained successfully.")
        st.metric("Accuracy", round(result["accuracy"], 3))
        report_df = pd.DataFrame(result["classification_report"]).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)
        st.subheader("Confusion Matrix")
        matrix_df = pd.DataFrame(
            result["confusion_matrix"],
            index=result["labels"],
            columns=result["labels"],
        )
        st.dataframe(matrix_df, use_container_width=True)
        save_model(result["model"])
        st.success("Model saved.")

st.divider()
st.subheader("Predict Error Category for a Segment")

model = load_model()

if model is None:
    st.warning("No saved model found. Train the model first.")
else:
    segment_ids = get_segment_ids()
    if len(segment_ids) == 0:
        st.warning("No segments available.")
    else:
        selected_segment_id = st.selectbox("Select Segment", segment_ids)
        segment = get_segment_by_id(selected_segment_id)
        if segment:
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

            if st.button("Predict Error Category"):
                prediction = predict_error_category(
                    model,
                    segment["source_text"],
                    segment["mt_output"],
                    segment["post_edited_text"],
                )
                st.success(f"Predicted category: {prediction['predicted_category']}")
                st.metric("Confidence", prediction["confidence"])
                scores_df = pd.DataFrame(
                    prediction["all_scores"].items(), columns=["Category", "Score"]
                )
                st.dataframe(scores_df, use_container_width=True)
