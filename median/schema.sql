-- Median game - Supabase schema
-- Run this in your Supabase SQL editor

create table if not exists answers (
  id          bigint generated always as identity primary key,
  day         date        not null,
  question_index integer   not null check (question_index between 0 and 4),
  value       numeric     not null,
  device_id   text        not null,
  is_seed     boolean     not null default false,
  created_at  timestamptz not null default now()
);

-- One real answer per device per question per day
create unique index if not exists answers_unique_real
  on answers (day, question_index, device_id)
  where not is_seed;

-- Fast reads for today's answers
create index if not exists answers_day_qi
  on answers (day, question_index);

-- Row Level Security
-- SELECT and INSERT are open to anon; UPDATE and DELETE have no policy so they
-- are denied by default (RLS denies anything without a matching policy).
alter table answers enable row level security;

create policy "public read" on answers
  for select using (true);

-- Reject is_seed = true from clients; seed answers are generated client-side
-- only and must never be written to the database.
create policy "public insert" on answers
  for insert with check (is_seed = false);
