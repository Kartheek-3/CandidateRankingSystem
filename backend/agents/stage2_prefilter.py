"""
Stage 2 — Fast Pre-Filter (Retrieval, not reasoning)
=====================================================
Embeds the RequirementSpec summary text (NOT the raw JD — the spec is cleaner,
more structured, and embeds better) and retrieves the top-K semantically similar
candidates from the FAISS index in milliseconds.

This keeps the expensive LLM reasoning in Stages 3 & 4 from running on every
candidate — we funnel from ~50k+ candidates down to a manageable shortlist first.

Outputs: list of dicts with {candidate_id, semantic_score, metadata_row}
"""

from __future__ import annotations
import logging
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from .stage1_jd_decomposer import RequirementSpec

logger = logging.getLogger(__name__)

# Singleton model — loaded once, reused across all requests
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info('[Stage2] Loading SentenceTransformer model...')
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


from supabase import create_client, Client
import os

_supabase_client: Client | None = None

def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        if url and key:
            _supabase_client = create_client(url, key)
    return _supabase_client


def retrieve(
    spec: RequirementSpec,
    index: faiss.Index,
    df: pd.DataFrame,
    top_k: int = 500,
    shortlist_ratio: float = 0.25,
    batch_id: str | None = None,
) -> list[dict]:
    """
    Embed the spec summary and retrieve the top-K candidates from Supabase pgvector.

    Args:
        spec:            RequirementSpec from Stage 1
        index:           Deprecated FAISS Index (kept for signature compatibility)
        df:              Deprecated parquet metadata DataFrame (kept for signature compatibility)
        top_k:           How many candidates to pull from FAISS initially
        shortlist_ratio: Fraction to keep for Stage 3 (default 25%)

    Returns:
        List of candidate dicts sorted by semantic_score descending.
        Each dict: {candidate_id, semantic_score, metadata}
    """
    model = _get_model()
    supabase = _get_supabase()
    
    if not supabase:
        raise RuntimeError("Supabase client not configured. Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")

    # Embed the structured spec summary — it's a dense, clean query
    query_text = spec.summary_for_embedding
    logger.info('[Stage2] Embedding query for Supabase: "%s"', query_text[:80])

    query_vec = model.encode([query_text], convert_to_numpy=True)[0].tolist()

    logger.info('[Stage2] Executing pgvector search via Supabase RPC...')
    
    rpc_payload = {
        'query_embedding': query_vec,
        'match_threshold': -1.0,
        'match_count': top_k,
        'filter_batch_id': batch_id if batch_id else None
    }
        
    response = supabase.rpc('match_candidates', rpc_payload).execute()
    
    data = response.data
    results = []
    for row in data:
        results.append({
            'candidate_id':   row['candidate_id'],
            'semantic_score': float(row['similarity']),
            'metadata':       row['metadata'],
        })

    # Sort descending by semantic score
    results.sort(key=lambda x: -x['semantic_score'])

    # Apply shortlist cap — keep top N% or at least 50, at most 300
    if len(results) > 0:
        shortlist_n = max(50, min(300, int(len(results) * shortlist_ratio)))
        shortlist   = results[:shortlist_n]
    else:
        shortlist = []

    logger.info('[Stage2] Retrieved %d candidates from Supabase, shortlisted to %d.',
                len(results), len(shortlist))
    return shortlist


# ---------------------------------------------------------------------------
# Utility — build candidate text for embedding (same as precompute.py logic)
# ---------------------------------------------------------------------------

def build_candidate_text(raw: dict) -> str:
    """
    Create a dense string representing the candidate's core qualifications.
    Used if we ever need to re-embed a candidate on-the-fly.
    """
    profile = raw.get('profile', {})
    skills  = ', '.join([
        s['name'] for s in raw.get('skills', [])
        if s.get('proficiency') in ['advanced', 'expert']
    ])
    career_parts = []
    for job in raw.get('career_history', []):
        career_parts.append(
            f"{job.get('title')} at {job.get('company')} "
            f"({job.get('duration_months')} months): {job.get('description', '')}"
        )
    career_text = ' | '.join(career_parts)

    return (
        f"Title: {profile.get('current_title')}. "
        f"Experience: {profile.get('years_of_experience')} years. "
        f"Summary: {profile.get('summary')} "
        f"Key Skills: {skills}. "
        f"Career History: {career_text}"
    )
