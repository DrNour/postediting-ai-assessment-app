-- EduApp Supabase schema
-- Run this once in the Supabase SQL Editor for a new project.
-- The Streamlit app connects from the server. Keep SUPABASE_KEY in Streamlit Secrets.

create extension if not exists pgcrypto;

create table if not exists public.assignments (
    assignment_id text primary key default gen_random_uuid()::text,
    course text,
    title text not null,
    instructions text,
    source_text text not null,
    machine_translation text,
    reference_translation text,
    due_date date,
    max_score double precision default 100,
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create table if not exists public.submissions (
    submission_id text primary key default gen_random_uuid()::text,
    assignment_id text,
    assignment_code text,
    task_id text,
    assignment_title text,
    task_type text not null default 'post_editing'
        check (task_type in ('translation', 'post_editing')),

    student_id text not null,
    student_name text,
    group_name text,
    semester text,

    source_language text,
    target_language text,
    domain text,

    source_text text,
    machine_translation text,
    reference_translation text,
    post_edited_text text not null,
    student_reflection text,
    editing_time_seconds double precision,

    inserted_words integer,
    deleted_words integer,
    replaced_segments integer,
    unchanged_words integer,
    source_word_count integer,
    mt_word_count integer,
    pe_word_count integer,

    mt_pe_cosine_similarity double precision,
    mt_pe_cosine_method text,
    mt_pe_edit_distance_ratio double precision,
    mt_pe_length_ratio double precision,
    mt_pe_lexical_similarity double precision,
    mt_pe_change_ratio double precision,
    mt_pe_bleu double precision,
    mt_pe_chrf double precision,
    mt_pe_ter double precision,
    mt_pe_bertscore_f1 double precision,

    pe_reference_cosine_similarity double precision,
    pe_reference_cosine_method text,
    pe_reference_length_ratio double precision,
    pe_reference_bleu double precision,
    pe_reference_chrf double precision,
    pe_reference_ter double precision,
    pe_reference_bertscore_f1 double precision,

    research_mode boolean default true,
    advanced_metrics_status text,
    raw_mt_word_count integer,
    ht_word_count integer,
    reference_word_count integer,
    mt_pe_word_count_difference integer,
    mt_pe_inserted_words integer,
    mt_pe_deleted_words integer,
    mt_pe_replaced_words integer,
    mt_pe_replacement_output_words integer,
    mt_pe_unchanged_words integer,
    mt_pe_changed_original_words integer,
    mt_pe_unchanged_ratio double precision,
    mt_pe_changed_ratio_original double precision,
    mt_pe_overlap_bleu double precision,
    mt_pe_overlap_chrf double precision,
    mt_pe_overlap_ter double precision,

    raw_mt_quality_bleu double precision,
    raw_mt_quality_chrf double precision,
    raw_mt_quality_ter double precision,
    pe_quality_bleu double precision,
    pe_quality_chrf double precision,
    pe_quality_ter double precision,
    ht_quality_bleu double precision,
    ht_quality_chrf double precision,
    ht_quality_ter double precision,
    raw_mt_quality_bertscore_f1 double precision,
    pe_quality_bertscore_f1 double precision,
    ht_quality_bertscore_f1 double precision,
    raw_mt_quality_comet double precision,
    pe_quality_comet double precision,
    ht_quality_comet double precision,

    mt_pe_interpretation text,
    quality_warnings text,
    teacher_score double precision,
    teacher_feedback text,
    submitted_at timestamptz not null default now()
);

create table if not exists public.teacher_annotations (
    id bigint generated always as identity primary key,
    annotation_id text,
    submission_id text not null,
    assignment_id text,
    assignment_title text,
    student_id text,
    student_name text,
    category text,
    subcategory text,
    severity text,
    selected_text text,
    teacher_comment text,
    suggested_revision text,
    annotator_id text,
    created_at timestamptz not null default now()
);

create table if not exists public.ai_feedback (
    id bigint generated always as identity primary key,
    feedback_id text not null,
    submission_id text not null,
    assignment_id text,
    assignment_title text,
    student_id text,
    student_name text,
    model_name text,
    ai_overall_comment text,
    ai_possible_errors jsonb,
    ai_rubric_scores jsonb,
    ai_raw_json jsonb,
    ai_risk_level text,
    ai_risk_warnings jsonb,
    teacher_review_status text default 'pending',
    teacher_review_comment text,
    created_at timestamptz not null default now()
);

create table if not exists public.ai_feedback_reviews (
    id bigint generated always as identity primary key,
    review_id text,
    feedback_id text not null,
    submission_id text not null,
    assignment_id text,
    assignment_title text,
    student_id text,
    student_name text,
    review_status text,
    teacher_final_feedback text,
    teacher_notes text,
    reviewer_id text,
    usefulness_rating integer check (
        usefulness_rating is null or usefulness_rating between 1 and 5
    ),
    teacher_review_time_seconds double precision,
    created_at timestamptz not null default now()
);

create table if not exists public.app_metric_settings (
    id text primary key default 'default',
    research_mode boolean not null default true,
    run_advanced_metrics_now boolean not null default false,
    show_student_metrics boolean not null default true,
    show_editing_summary boolean not null default true,
    show_mt_pe_overlap_metrics boolean not null default true,
    show_reference_quality_metrics boolean not null default true,
    show_automated_interpretation boolean not null default true,
    use_bert boolean not null default false,
    bert_language text not null default 'en',
    use_comet boolean not null default false,
    use_llm_judge boolean not null default false,
    updated_at timestamptz not null default now()
);

insert into public.app_metric_settings (id)
values ('default')
on conflict (id) do nothing;

create index if not exists idx_assignments_active
    on public.assignments (active, created_at desc);
create index if not exists idx_submissions_assignment
    on public.submissions (assignment_title, submitted_at desc);
create index if not exists idx_submissions_task_type
    on public.submissions (task_type, submitted_at desc);
create index if not exists idx_submissions_student
    on public.submissions (student_id, submitted_at desc);
create index if not exists idx_annotations_submission
    on public.teacher_annotations (submission_id, created_at desc);
create index if not exists idx_ai_feedback_submission
    on public.ai_feedback (submission_id, created_at desc);
create index if not exists idx_ai_reviews_submission
    on public.ai_feedback_reviews (submission_id, created_at desc);

-- No public database policies are created here. The app is designed to use a
-- server-side key stored in Streamlit Secrets. RLS prevents direct anonymous
-- browser access while the server-side key performs the required operations.
alter table public.assignments enable row level security;
alter table public.submissions enable row level security;
alter table public.teacher_annotations enable row level security;
alter table public.ai_feedback enable row level security;
alter table public.ai_feedback_reviews enable row level security;
alter table public.app_metric_settings enable row level security;
