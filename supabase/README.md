# Supabase setup

For a new Supabase project, run `schema.sql` once in the SQL Editor.

For an existing EduApp project, run `migrations/001_add_task_type.sql` once. It adds the `task_type` column and marks legacy records as `post_editing`.

The schema enables Row Level Security but creates no anonymous policies. Put a **server-side Supabase key** in Streamlit Secrets as `SUPABASE_KEY`; never place that key in browser-side code or commit it to GitHub. All teacher and research pages are also protected by `TEACHER_PASSWORD` inside the app.
