# AI-Assisted Post-Editing Assessment Tool

This is a Streamlit-based research app for analysing machine translation post-editing performance in Arabic-English translator training.

The app supports student submissions, editing-time tracking, teacher error annotation, MT-vs-post-edited comparison metrics, semantic similarity scoring, error classification, AI-generated feedback, teacher approval workflows, and research evaluation exports.

## Core Principle

**AI suggests. Teacher decides.**

AI-generated feedback is draft feedback only. It must be reviewed by a teacher before being used for student assessment.

## Features

- SQLite database for storing student submissions
- Student submission page
- Teacher annotation page
- Translation error taxonomy
- Dashboard with editing-time and annotation summaries
- MT vs post-edited comparison metrics
- Semantic similarity scoring with sentence-transformers
- Baseline error classification model using TF-IDF + Logistic Regression
- GenAI feedback generator using a taxonomy-based prompt
- Teacher approval/edit/reject workflow
- Research evaluation dashboard
- CSV export for dissertation analysis

## Folder Structure

```text
postediting-ai-assessment-app/
├── app.py
├── database.py
├── requirements.txt
├── README.md
├── .gitignore
├── modules/
│   ├── taxonomy.py
│   ├── metrics.py
│   ├── similarity.py
│   ├── error_classifier.py
│   ├── ai_feedback.py
│   └── evaluation.py
├── pages/
│   ├── 1_Student_Submission.py
│   ├── 2_Teacher_Annotation.py
│   ├── 3_Dashboard.py
│   ├── 4_Error_Classifier.py
│   ├── 5_AI_Feedback.py
│   ├── 6_Teacher_Review.py
│   └── 7_Evaluation.py
├── sample_data/
│   └── demo_segments.csv
├── scripts/
│   └── load_demo_data.py
└── docs/
    ├── taxonomy.md
    ├── rubric.md
    └── pilot_study_design.md
```

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/postediting-ai-assessment-app.git
cd postediting-ai-assessment-app
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate it:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Optional: load demo data:

```bash
python scripts/load_demo_data.py
```

6. Run the app:

```bash
streamlit run app.py
```

## API Key Setup

For GenAI feedback, add your OpenAI API key.

Local option using `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

Streamlit Cloud option:

Use Streamlit secrets and add:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Do not commit real API keys to GitHub.

## Pages

### 1. Student Submission

Save source text, machine translation, student post-edited version, and editing time.

### 2. Teacher Annotation

Label segments using the translation error taxonomy.

### 3. Dashboard

View saved records, editing-time summaries, category counts, severity counts, and teacher review summaries.

### 4. Error Classifier

Train a baseline classifier from teacher-labelled data and predict error categories for new segments.

### 5. AI Feedback

Generate draft AI feedback using the taxonomy, rubric, and automatic metrics.

### 6. Teacher Review

Approve, edit, reject, and save final teacher-reviewed feedback.

### 7. Evaluation

Export research data and calculate approval rates, AI-teacher agreement, and usefulness summaries.

## Privacy Note

The demo dataset is synthetic/anonymised. Real student submissions should not be uploaded to a public version of the app. For classroom or research use, obtain ethical approval, informed consent, and secure the data.

## Suggested Research Questions

1. To what extent do AI-generated error categories agree with teacher-assigned categories?
2. How often do teachers approve, edit, or reject AI-generated feedback?
3. How useful do teachers perceive AI-generated feedback to be?
4. What types of AI feedback require the most teacher editing?

## Suggested CV Bullet

Developed a Python/Streamlit app for AI-assisted post-editing assessment, integrating SQLite data storage, teacher annotation, MT-vs-post-edited comparison metrics, semantic similarity scoring, GenAI feedback, and teacher review workflows.

## Limitations

- The baseline error classifier requires enough labelled examples.
- Semantic similarity scores are warning signals, not final quality judgements.
- AI feedback can be wrong or incomplete and requires teacher review.
- Public deployment should use synthetic or anonymised data only.
