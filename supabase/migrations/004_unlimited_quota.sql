-- Migration 004: Enable unlimited research papers (remove 5-paper quota restriction)
-- Run this in Supabase SQL Editor if you previously applied 002/003 migrations.

-- Update function to always grant quota
create or replace function public.check_user_quota(p_user_id uuid)
returns boolean
language sql security definer set search_path = public
as $$
  select true;
$$;

grant execute on function public.check_user_quota(uuid) to service_role;
