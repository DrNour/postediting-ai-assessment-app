import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/error_classifier.joblib")


def create_model_input(row):
    source = str(row.get("source_text", ""))
    mt = str(row.get("mt_output", ""))
    pe = str(row.get("post_edited_text", ""))
    return f"Source: {source} MT: {mt} PE: {pe}"


def prepare_training_data(df):
    labelled_df = df.dropna(subset=["category"]).copy()
    labelled_df = labelled_df[labelled_df["category"].astype(str).str.len() > 0]
    labelled_df["model_input"] = labelled_df.apply(create_model_input, axis=1)
    X = labelled_df["model_input"]
    y = labelled_df["category"]
    return X, y, labelled_df


def build_classifier():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            ),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train_and_evaluate_classifier(df):
    X, y, labelled_df = prepare_training_data(df)

    if len(labelled_df) < 10:
        return {"error": "Not enough labelled data. Add at least 10 annotated segments first."}

    if y.nunique() < 2:
        return {"error": "You need at least two different error categories to train the model."}

    min_class_count = y.value_counts().min()
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = build_classifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y.unique())
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    accuracy = accuracy_score(y_test, predictions)

    return {
        "model": model,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": labels,
        "test_examples": X_test,
        "true_labels": y_test,
        "predicted_labels": predictions,
    }


def save_model(model):
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def predict_error_category(model, source_text, mt_output, post_edited_text):
    input_text = f"Source: {source_text} MT: {mt_output} PE: {post_edited_text}"
    prediction = model.predict([input_text])[0]

    confidence_scores = {}
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([input_text])[0]
        classes = model.classes_
        confidence_scores = {
            label: round(float(prob), 3) for label, prob in zip(classes, probabilities)
        }
        confidence = max(confidence_scores.values())

    return {
        "predicted_category": prediction,
        "confidence": confidence,
        "all_scores": confidence_scores,
    }
