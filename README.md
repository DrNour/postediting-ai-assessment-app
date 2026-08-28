# EduApp: Translation and Post-Editing Assessment

EduApp is a Streamlit application for Arabic–English translator training and research. Students can now complete either of two task types:

- **Translation:** translate independently from the source text.
- **Post-editing:** revise a supplied machine translation.

The app keeps these modes methodologically separate. A translation task does not show the MT output to the student and does not calculate misleading MT–PE effort measures. Post-editing tasks retain track changes, edit-operation counts, and MT–PE effort metrics.

> **Core principle:** AI suggests. The teacher decides.

## What was fixed

The project previously mixed Streamlit's automatic `pages/` navigation with a second sidebar inside `app.py`. That produced the bare **app** item and a confusing page list. The fixed version uses one `st.navigation` router in `app.py`, with grouped Student, Teacher, AI, and Research pages.

Other fixes include:

- one clean project root; duplicate nested app folders were removed;
- a student choice between Translation and Post-editing;
- task-aware labels, prompts, metrics, annotation views, exports, dashboards, and evaluation data;
- a `task_type` field saved with every submission;
- teacher-password protection for all teacher, AI, dashboard, and research pages;
- lazy semantic-model loading so the AI Feedback page does not appear frozen;
- complete Supabase setup files and a usable secrets template;
- a lighter core dependency list, with neural metrics moved to an optional requirements file;
- theme-safe track-change styling.

## Project structure

```text
.
├── app.py                         # Single Streamlit navigation router
├── workflow_pages.py              # Assignment/student/teacher workflows
├── views/
│   ├── Home.py
│   ├── Student_Assignments.py
│   ├── Teacher_Assignments.py
│   └── Teacher_Submissions.py
├── pages/
│   ├── 1_Student_Submission.py    # Manual/pilot entry
│   ├── 2_Teacher_Annotation.py
│   ├── 3_Dashboard.py
│   ├── 4_Error_Classifier.py
│   ├── 5_AI_Feedback.py
│   ├── 6_Research_Analytics.py
│   ├── 7_Evaluation.py
│   └── 8_Teacher_Review.py
├── modules/
│   ├── auth.py
│   ├── config.py
│   ├── task_mode.py
│   └── ...
├── supabase/
│   ├── schema.sql
│   └── migrations/001_add_task_type.sql
├── requirements.txt
├── requirements-advanced.txt
└── .streamlit/secrets.toml.example
```

`database.py` and `scripts/load_demo_data.py` are legacy SQLite utilities from the older prototype. The current Streamlit application uses Supabase.

## 1. Install

Python 3.11 is recommended and specified in `runtime.txt`.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install the working core application:

```bash
pip install -r requirements.txt
```

Neural semantic similarity and BERTScore are optional because they download large models. Install them only when needed:

```bash
pip install -r requirements-advanced.txt
```

## 2. Configure Supabase

### New Supabase project

Open the Supabase SQL Editor and run:

```text
supabase/schema.sql
```

The schema creates these tables:

- `assignments`
- `submissions`
- `teacher_annotations`
- `ai_feedback`
- `ai_feedback_reviews`
- `app_metric_settings`

### Existing EduApp Supabase project

Run this migration once:

```text
supabase/migrations/001_add_task_type.sql
```

It adds `task_type`, labels legacy rows as `post_editing`, and creates the task-type constraint and index.

The included schema enables Row Level Security and does not create anonymous public policies. Use a **server-side Supabase key** in Streamlit Secrets. Never commit or display that key.

## 3. Configure Streamlit Secrets

Copy the example file:

```bash
# macOS/Linux
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Windows PowerShell
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then add real values:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "your_supabase_server_key"
TEACHER_PASSWORD = "a-long-random-password"
OPENAI_API_KEY = "optional-openai-key"
```

For Streamlit Community Cloud, paste the same TOML into **App settings → Secrets**. Do not upload the real `secrets.toml` file to GitHub.

## 4. Run

From the project root:

```bash
streamlit run app.py
```

Do not run an individual file inside `pages/`. `app.py` is the router and must be the Streamlit entry point.

## Student workflow

### Assignment with an MT output

The student sees a task selector:

1. **Translate from the source text**
2. **Post-edit the machine translation**

In Translation mode, the MT is hidden, the response box starts blank, and only task-appropriate reference-based metrics are used.

In Post-editing mode, the raw MT is visible, the response box begins with the MT text, and the app shows track changes and editing-effort indicators.

### Assignment without an MT output

Only Translation mode is offered.

The final student output remains stored in the legacy `post_edited_text` column for compatibility with existing annotation and analytics code. The new `task_type` column tells every page whether that output is a translation or a post-edited text.

## Teacher workflow

Teacher pages use the shared `TEACHER_PASSWORD` login and preserve the login across pages during the session.

Teachers can:

- create assignments with source text, optional MT, and optional reference translation;
- review and score submissions by assignment and task type;
- annotate errors with the taxonomy;
- generate AI feedback drafts and approve, edit, or reject them;
- export task-aware CSV, Excel, and Word data;
- analyse translation and post-editing records separately.

## AI Feedback page

The page no longer loads a sentence-transformer model automatically. Semantic similarity runs only when **Calculate semantic similarity now** is selected. This prevents the page from looking blank or stuck during first model download.

`OPENAI_API_KEY` is optional for the rest of the app. It is required only when generating AI feedback.

## Metric interpretation

- **MT–PE metrics** describe overlap and editing effort. They are used only for post-editing records.
- **Reference-based metrics** such as BLEU, chrF, TER, BERTScore, or COMET can be used for either task type when an independent reference exists.
- Automatic metrics are indicators, not final grades.
- Teacher scores, annotations, and reviewed feedback remain the human assessment layer.

## Troubleshooting

### The sidebar still shows “app” and the old flat list

Confirm that the deployment points to the fixed repository root and runs `app.py`. Also reinstall from the updated `requirements.txt`, which requires Streamlit 1.36 or later.

### A page says Supabase is not configured

Add `SUPABASE_URL` and `SUPABASE_KEY` to Streamlit Secrets. The Home page remains visible even before configuration and reports missing keys.

### Submission fails with a `task_type` column error

Run:

```text
supabase/migrations/001_add_task_type.sql
```

Then submit again.

### AI Feedback looks slow

Leave semantic similarity unchecked unless the optional neural packages and model are needed. The first model load can be large.

### Teacher pages are locked

Set `TEACHER_PASSWORD` in Streamlit Secrets, then use the sidebar login.

## Privacy and research use

Use anonymised student identifiers, obtain the required ethical approvals and consent, and do not expose real student records through permissive public database policies. AI-generated feedback must be reviewed by a teacher before assessment use.
