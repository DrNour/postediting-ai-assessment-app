-- Existing projects: add the dual-mode task field without losing old records.

alter table public.submissions
    add column if not exists task_type text;

update public.submissions
set task_type = case
    when lower(replace(coalesce(task_type, ''), '-', '_')) = 'translation'
        then 'translation'
    when lower(replace(coalesce(task_type, ''), '-', '_')) in
        ('post_editing', 'postediting', 'post_edit', 'pe')
        then 'post_editing'
    else 'post_editing'
end;

alter table public.submissions
    alter column task_type set default 'post_editing';

alter table public.submissions
    alter column task_type set not null;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'submissions_task_type_check'
          and conrelid = 'public.submissions'::regclass
    ) then
        alter table public.submissions
            add constraint submissions_task_type_check
            check (task_type in ('translation', 'post_editing'));
    end if;
end $$;

create index if not exists idx_submissions_task_type
    on public.submissions (task_type, submitted_at desc);
