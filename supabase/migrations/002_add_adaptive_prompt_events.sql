-- Store a trace of human-authored MQM prompt use with each submitted task.
-- Apply once in the Supabase SQL Editor before deploying the matching app code.
ALTER TABLE public.submissions
ADD COLUMN IF NOT EXISTS adaptive_prompt_events JSONB NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN public.submissions.adaptive_prompt_events IS
'MQM-informed adaptive prompt events recorded during the task; contains category, timing, learner question, and minimised interaction metadata.';
