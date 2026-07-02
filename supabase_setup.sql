-- Enable vector extension
create extension if not exists vector;

-- Create candidates table (if not exists)
create table if not exists candidates_vector (
  id uuid primary key default uuid_generate_v4(),
  candidate_id text not null unique,
  metadata jsonb not null,
  embedding vector(384)
);

-- Explicitly drop BOTH possible old signatures to avoid "not unique" errors
drop function if exists match_candidates(vector(384), float, int);
drop function if exists match_candidates(vector(384), float, int, text);

-- Create the new matching function with batch_id support
create or replace function match_candidates(
  query_embedding vector(384),
  match_threshold float,
  match_count int,
  filter_batch_id text default null
)
returns table (
  id uuid,
  candidate_id text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    candidates_vector.id,
    candidates_vector.candidate_id,
    candidates_vector.metadata,
    1 - (candidates_vector.embedding <=> query_embedding) as similarity
  from candidates_vector
  where 
    1 - (candidates_vector.embedding <=> query_embedding) > match_threshold
    and (
      (filter_batch_id is null and candidates_vector.metadata->>'batch_id' is null)
      or 
      (candidates_vector.metadata->>'batch_id' = filter_batch_id)
    )
  order by candidates_vector.embedding <=> query_embedding
  limit match_count;
$$;

-- Force Supabase cache to reload immediately
NOTIFY pgrst, 'reload schema';
