import pandas as pd
import streamlit as st
from supabase import create_client

from modules.auth import require_teacher_access
from modules.task_mode import POST_EDITING, is_translation, normalize_task_type, student_output_label, task_type_label

from modules.error_classifier import (
    load_model,
    predict_error_category,
    save_model,
    train_and_evaluate_classifier,
)


st.title("Translation Error Classification Model")
st.write(
    "Train a baseline machine-learning model using teacher-labelled error categories."
)


if not require_teacher_access("error_classifier"):
    st.stop()


# ============================================================
# Supabase connection
# ============================================================

@st.cache_resource
def get_supabase_client():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        st.error(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit Secrets."
        )
        st.stop()


supabase = get_supabase_client()


# ============================================================
# Helper functions
# ============================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def load_submissions():
    try:
        response = (
            supabase.table("submissions")
            .select("*")
            .order("submitted_at", desc=True)
            .execute()
        )

        return pd.DataFrame(response.data or [])

    except Exception as error:
        st.error("Could not load submissions from Supabase.")
        st.code(str(error))
        return pd.DataFrame()


def load_annotations():
    try:
        response = (
            supabase.table("teacher_annotations")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        return pd.DataFrame(response.data or [])

    except Exception as error:
        st.error("Could not load teacher annotations from Supabase.")
        st.code(str(error))
        return pd.DataFrame()


def load_training_data():
    """
    Builds a classifier-ready dataframe by merging:

    submissions:
        source_text, machine_translation, post_edited_text

    teacher_annotations:
        category, subcategory, severity, teacher_comment, suggested_revision

    The classifier module expects:
        segment_id, source_text, mt_output, post_edited_text, category
    """

    submissions = load_submissions()
    annotations = load_annotations()

    if submissions.empty or annotations.empty:
        return pd.DataFrame()

    if "submission_id" not in submissions.columns:
        st.error("The submissions table does not contain submission_id.")
        return pd.DataFrame()

    if "submission_id" not in annotations.columns:
        st.error("The teacher_annotations table does not contain submission_id.")
        return pd.DataFrame()

    merged = annotations.merge(
        submissions,
        on="submission_id",
        how="left",
        suffixes=("_annotation", "_submission"),
    )

    if merged.empty:
        return pd.DataFrame()

    training_df = pd.DataFrame()

    training_df["segment_id"] = merged["submission_id"].astype(str)

    training_df["source_text"] = merged.get(
        "source_text",
        pd.Series([""] * len(merged)),
    )

    training_df["mt_output"] = merged.get(
        "machine_translation",
        pd.Series([""] * len(merged)),
    )

    training_df["post_edited_text"] = merged.get(
        "post_edited_text",
        pd.Series([""] * len(merged)),
    )

    training_df["task_type"] = merged.get(
        "task_type",
        pd.Series([POST_EDITING] * len(merged)),
    ).apply(normalize_task_type)

    training_df["category"] = merged.get(
        "category",
        pd.Series([None] * len(merged)),
    )

    training_df["subcategory"] = merged.get(
        "subcategory",
        pd.Series([None] * len(merged)),
    )

    training_df["severity"] = merged.get(
        "severity",
        pd.Series([None] * len(merged)),
    )

    training_df["teacher_comment"] = merged.get(
        "teacher_comment",
        pd.Series([""] * len(merged)),
    )

    training_df["suggested_revision"] = merged.get(
        "suggested_revision",
        pd.Series([""] * len(merged)),
    )

    training_df["selected_text"] = merged.get(
        "selected_text",
        pd.Series([""] * len(merged)),
    )

    return training_df


def load_prediction_submissions():
    """
    Loads unmerged submissions for prediction.
    """

    submissions = load_submissions()

    if submissions.empty:
        return pd.DataFrame()

    return submissions


# ============================================================
# Load labelled training data
# ============================================================

df = load_training_data()

if df.empty:
    labelled_df = pd.DataFrame()
else:
    labelled_df = df.dropna(subset=["category"])
    labelled_df = labelled_df[labelled_df["category"].astype(str).str.strip() != ""]


# ============================================================
# Labelled dataset display
# ============================================================

st.subheader("Labelled Dataset")
st.metric("Labelled Examples", len(labelled_df))

if len(labelled_df) > 0:
    display_columns = [
        "segment_id",
        "task_type",
        "source_text",
        "mt_output",
        "post_edited_text",
        "category",
        "subcategory",
        "severity",
        "selected_text",
        "teacher_comment",
    ]

    available_columns = [
        column for column in display_columns if column in labelled_df.columns
    ]

    st.dataframe(
        labelled_df[available_columns],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No labelled annotations found yet. Add teacher annotations first, "
        "then return to train the classifier."
    )


# ============================================================
# Train model
# ============================================================

st.divider()
st.subheader("Train Model")

st.warning(
    "This is a baseline classifier. It is useful for experimentation, but it should not "
    "replace teacher judgement."
)

if st.button("Train Error Classifier"):
    if len(labelled_df) < 2:
        st.error(
            "You need at least 2 labelled examples to train a basic classifier. "
            "For meaningful research use, you need many more."
        )
    elif labelled_df["category"].nunique() < 2:
        st.error(
            "You need annotations from at least 2 different error categories "
            "to train and evaluate a classifier."
        )
    else:
        result = train_and_evaluate_classifier(labelled_df)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Model trained successfully.")

            st.metric("Accuracy", round(result["accuracy"], 3))

            report_df = pd.DataFrame(
                result["classification_report"]
            ).transpose()

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


# ============================================================
# Predict error category
# ============================================================

st.divider()
st.subheader("Predict Error Category for a Submission")

model = load_model()

if model is None:
    st.warning("No saved model found. Train the model first.")

else:
    submissions = load_prediction_submissions()

    if submissions.empty:
        st.warning("No student submissions available.")

    else:
        submissions = submissions.copy()
        if "task_type" not in submissions.columns:
            submissions["task_type"] = POST_EDITING
        submissions["task_type"] = submissions["task_type"].apply(normalize_task_type)

        submission_labels = []

        for _, row in submissions.iterrows():
            label = (
                f"{safe_text(row.get('student_id'))} — "
                f"{safe_text(row.get('student_name'))} — "
                f"{task_type_label(row.get('task_type'))} — "
                f"{safe_text(row.get('assignment_title'))} — "
                f"ID {safe_text(row.get('submission_id'))}"
            )
            submission_labels.append(label)

        selected_label = st.selectbox(
            "Select submission",
            submission_labels,
        )

        selected_submission_id = selected_label.split("ID ")[-1].strip()

        selected_df = submissions[
            submissions["submission_id"].astype(str) == selected_submission_id
        ]

        if selected_df.empty:
            st.error("Could not find the selected submission.")
            st.stop()

        submission = selected_df.iloc[0].to_dict()

        selected_task_type = normalize_task_type(submission.get("task_type"))
        st.write(f"**Task type:** {task_type_label(selected_task_type)}")

        if is_translation(selected_task_type):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Source Text**")
                st.write(safe_text(submission.get("source_text")))
            with col2:
                st.markdown(f"**{student_output_label(selected_task_type)}**")
                st.write(safe_text(submission.get("post_edited_text")))
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Source Text**")
                st.write(safe_text(submission.get("source_text")))
            with col2:
                st.markdown("**Machine Translation**")
                st.write(safe_text(submission.get("machine_translation")))
            with col3:
                st.markdown(f"**{student_output_label(selected_task_type)}**")
                st.write(safe_text(submission.get("post_edited_text")))

        if st.button("Predict Error Category"):
            prediction = predict_error_category(
                model,
                safe_text(submission.get("source_text")),
                safe_text(submission.get("machine_translation")),
                safe_text(submission.get("post_edited_text")),
            )

            st.success(
                f"Predicted category: {prediction['predicted_category']}"
            )

            st.metric(
                "Confidence",
                prediction["confidence"],
            )

            scores_df = pd.DataFrame(
                prediction["all_scores"].items(),
                columns=["Category", "Score"],
            )

            st.dataframe(
                scores_df,
                use_container_width=True,
                hide_index=True,
            )
