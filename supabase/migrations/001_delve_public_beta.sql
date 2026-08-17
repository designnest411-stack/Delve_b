-- Delve public-beta schema. Run in Supabase SQL Editor before deploying.
create extension if not exists vector;

create table if not exists public.research_sessions (
  id uuid primary key,
  owner_id uuid not null references auth.users(id) on delete cascade,
  topic text not null check (char_length(topic) between 1 and 500),
  status text not null default 'initializing' check (status in ('initializing', 'queued', 'running', 'complete', 'error', 'cancelled')),
  current_step text not null default 'initializing',
  controls jsonb not null default '{}'::jsonb,
  uploaded_paper_ids jsonb not null default '[]'::jsonb,
  timeline jsonb not null default '[]'::jsonb,
  result jsonb not null default '{}'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  chat_history jsonb not null default '[]'::jsonb,
  resource_files jsonb not null default '[]'::jsonb,
  error text,
  started_at timestamptz,
  finished_at timestamptz,
  updated_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

create index if not exists research_sessions_owner_updated_idx
  on public.research_sessions(owner_id, updated_at desc);

create table if not exists public.research_jobs (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.research_sessions(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  kind text not null default 'research' check (kind in ('research', 'document')),
  status text not null default 'queued' check (status in ('queued', 'running', 'complete', 'error', 'cancelled')),
  attempts integer not null default 0,
  error text,
  qstash_message_id text,
  locked_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists research_jobs_session_idx on public.research_jobs(session_id);
create index if not exists research_jobs_owner_status_idx on public.research_jobs(owner_id, status);

-- Atomic and idempotent: duplicate QStash deliveries receive no row after the
-- first worker has changed a queued job to running.
create or replace function public.claim_research_job(p_job_id uuid)
returns setof public.research_jobs
language sql
security definer
set search_path = public
as $$
  update public.research_jobs
  set status = 'running', attempts = attempts + 1, locked_at = now(), updated_at = now()
  where id = p_job_id and status = 'queued'
  returning *;
$$;
revoke all on function public.claim_research_job(uuid) from public;
grant execute on function public.claim_research_job(uuid) to service_role;

create table if not exists public.research_documents (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid references public.research_sessions(id) on delete set null,
  filename text not null,
  storage_path text not null unique,
  size_bytes bigint not null check (size_bytes > 0),
  page_count integer,
  extracted_characters integer,
  status text not null default 'processing' check (status in ('processing', 'ready', 'error', 'deleted')),
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists research_documents_owner_idx on public.research_documents(owner_id, created_at desc);

create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.research_documents(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(384),
  created_at timestamptz not null default now(),
  unique(document_id, chunk_index)
);

create index if not exists document_chunks_owner_idx on public.document_chunks(owner_id);

create or replace function public.match_document_chunks(
  p_owner_id uuid,
  p_document_ids uuid[],
  p_query_embedding vector(384),
  p_match_count integer default 5
)
returns table (content text, metadata jsonb, distance double precision)
language sql stable security definer set search_path = public
as $$
  select dc.content, dc.metadata, (dc.embedding <=> p_query_embedding)::double precision as distance
  from public.document_chunks dc
  join public.research_documents d on d.id = dc.document_id
  where dc.owner_id = p_owner_id
    and dc.document_id = any(p_document_ids)
    and d.status = 'ready'
    and dc.embedding is not null
  order by dc.embedding <=> p_query_embedding
  limit greatest(1, least(p_match_count, 20));
$$;
revoke all on function public.match_document_chunks(uuid, uuid[], vector, integer) from public;
grant execute on function public.match_document_chunks(uuid, uuid[], vector, integer) to service_role;

create table if not exists public.llm_usage (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.research_sessions(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  agent text not null,
  model text not null,
  input_tokens integer not null default 0,
  output_tokens integer not null default 0,
  cache_read_tokens integer not null default 0,
  estimated_cost_usd numeric(12, 6) not null default 0,
  created_at timestamptz not null default now()
);

alter table public.research_sessions enable row level security;
alter table public.research_jobs enable row level security;
alter table public.research_documents enable row level security;
alter table public.document_chunks enable row level security;
alter table public.llm_usage enable row level security;

create policy "users manage own sessions" on public.research_sessions
  for all using (owner_id = auth.uid()) with check (owner_id = auth.uid());
create policy "users view own jobs" on public.research_jobs
  for select using (owner_id = auth.uid());
create policy "users manage own documents" on public.research_documents
  for all using (owner_id = auth.uid()) with check (owner_id = auth.uid());
create policy "users view own chunks" on public.document_chunks
  for select using (owner_id = auth.uid());
create policy "users view own usage" on public.llm_usage
  for select using (owner_id = auth.uid());

insert into storage.buckets (id, name, public)
values ('delve-documents', 'delve-documents', false)
on conflict (id) do update set public = false;

create policy "users upload own Delve files" on storage.objects
  for insert to authenticated
  with check (bucket_id = 'delve-documents' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "users read own Delve files" on storage.objects
  for select to authenticated
  using (bucket_id = 'delve-documents' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "users delete own Delve files" on storage.objects
  for delete to authenticated
  using (bucket_id = 'delve-documents' and (storage.foldername(name))[1] = auth.uid()::text);
