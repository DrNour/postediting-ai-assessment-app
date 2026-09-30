-- Allow the AI-assisted portfolio workflow in existing and new deployments.

alter table public.submissions
    drop constraint if exists submissions_task_type_check;

alter table public.submissions
    add constraint submissions_task_type_check
    check (task_type in ('translation', 'adaptive_translation', 'post_editing'));
