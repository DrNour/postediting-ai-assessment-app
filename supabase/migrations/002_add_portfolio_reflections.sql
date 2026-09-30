-- Portfolio reflection support for the three TRS430 workflow tasks.
-- Run this after supabase/schema.sql (or in the Supabase SQL Editor for an existing project).

create table if not exists public.portfolio_reflections (
    reflection_id text primary key default gen_random_uuid()::text,
    student_id text not null,
    student_name text,
    group_name text,
    semester text,
    assignment_code text not null,
    assignment_title text,
    task_type text not null check (
        task_type in ('translation', 'adaptive_translation', 'post_editing')
    ),
    source_url text,
    ai_tool text,
    reflection_text text not null,
    submitted_at timestamptz not null default now()
);

create index if not exists idx_portfolio_reflections_student
    on public.portfolio_reflections (student_id, submitted_at desc);

create index if not exists idx_portfolio_reflections_task
    on public.portfolio_reflections (task_type, submitted_at desc);

alter table public.portfolio_reflections enable row level security;
