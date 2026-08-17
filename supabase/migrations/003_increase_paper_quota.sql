-- Migration 003: Increase default lifetime paper limit from 1 to 5
-- Run this in Supabase SQL Editor after 002_lifetime_quota.sql

-- 1. Update the column default on user_paper_quotas table
alter table public.user_paper_quotas 
  alter column free_papers_allowed set default 5;

-- 2. Upgrade any existing user records to allow 5 free papers
update public.user_paper_quotas 
set free_papers_allowed = 5, updated_at = now() 
where free_papers_allowed < 5;
