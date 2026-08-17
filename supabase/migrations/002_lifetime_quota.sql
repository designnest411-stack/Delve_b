-- Migration 002: Add lifetime paper limit tracking
-- Run this in Supabase SQL Editor after 001_delve_public_beta.sql

-- Add a table to track user paper generation counts
create table if not exists public.user_paper_quotas (
  user_id uuid primary key references auth.users(id) on delete cascade,
  papers_generated integer not null default 0,
  free_papers_allowed integer not null default 5,
  last_paper_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.user_paper_quotas enable row level security;

create policy "users view own quota" on public.user_paper_quotas
  for select using (user_id = auth.uid());

-- Function to check if user has quota available
create or replace function public.check_user_quota(p_user_id uuid)
returns boolean
language plpgsql security definer set search_path = public
as $$
declare
  v_generated integer;
  v_allowed integer;
begin
  select coalesce(papers_generated, 0), free_papers_allowed
  into v_generated, v_allowed
  from public.user_paper_quotas
  where user_id = p_user_id;

  -- If no record exists, user hasn't generated any papers yet
  if not found then
    return true;
  end if;

  return v_generated < v_allowed;
end;
$$;

grant execute on function public.check_user_quota(uuid) to service_role;

-- Function to increment paper count when a paper completes
create or replace function public.increment_user_paper_count(p_user_id uuid)
returns void
language plpgsql security definer set search_path = public
as $$
begin
  insert into public.user_paper_quotas (user_id, papers_generated, last_paper_at, updated_at)
  values (p_user_id, 1, now(), now())
  on conflict (user_id) do update
  set papers_generated = user_paper_quotas.papers_generated + 1,
      last_paper_at = now(),
      updated_at = now();
end;
$$;

grant execute on function public.increment_user_paper_count(uuid) to service_role;

-- Trigger to auto-increment when a session completes
create or replace function public.track_completed_paper()
returns trigger
language plpgsql security definer set search_path = public
as $$
begin
  -- Only increment on transition to 'complete' status
  if NEW.status = 'complete' and (OLD.status is null or OLD.status != 'complete') then
    perform public.increment_user_paper_count(NEW.owner_id);
  end if;
  return NEW;
end;
$$;

drop trigger if exists track_completed_paper_trigger on public.research_sessions;
create trigger track_completed_paper_trigger
  after update of status on public.research_sessions
  for each row
  execute function public.track_completed_paper();
