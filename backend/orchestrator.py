"""
Orchestrator — Multi-Agent Pipeline Controller
===============================================
Sequences Stages 0-5, passes structured JSON between them, handles
retries, and falls back gracefully if any LLM stage fails.

Parallelism opportunity: Stage 3 (per-candidate evaluation) runs concurrently
via ThreadPoolExecutor inside stage3_evaluator.run_batch().

Data flow:
  JD text
    → Stage 1: RequirementSpec
    → Stage 2: Shortlisted candidates (FAISS)
    → [JSONL scan] Full profiles for top candidates
    → Stage 3: EvaluationResult[] (concurrent)
    → Stage 4: CriticResult[]
    → Stage 5: RankedCandidate[] (final output)

The orchestrator also maintains a `progress` dict so the Flask API can
report stage-by-stage status to the frontend.
"""

from __future__ import annotations
import json
import logging
import os
import sys
import time
from pathlib import Path

import faiss
import pandas as pd

# Add backend dir to path when run as CLI
sys.path.insert(0, str(Path(__file__).parent))

from agents.stage1_jd_decomposer import decompose, RequirementSpec
from agents.stage2_prefilter      import retrieve
from agents.stage3_evaluator      import run_batch as evaluate_batch
from agents.stage4_critic         import critique_batch
from agents.stage5_synthesizer    import synthesize, RankedCandidate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger('orchestrator')

# ---------------------------------------------------------------------------
# Global resource handles (loaded once on startup)
# ---------------------------------------------------------------------------

_faiss_index: faiss.Index | None = None
_metadata_df: pd.DataFrame | None = None


def load_resources(
    index_path: str | None = None,
    meta_path:  str | None = None,
) -> None:
    """Load resources (now points to Supabase, no local FAISS needed)."""
    global _faiss_index, _metadata_df
    logger.info('Running in Supabase pgvector mode. No local FAISS index needed.')
    _faiss_index = None
    _metadata_df = None


# ---------------------------------------------------------------------------
# Candidate full-profile loader
# ---------------------------------------------------------------------------

def _load_full_profiles(
    candidate_ids: set[str],
    jsonl_path: str,
    progress: dict,
) -> dict[str, dict]:
    """
    Scan the JSONL file and return full profiles for the given candidate IDs.
    Uses a fast string-contains pre-check before JSON parsing.
    """
    progress['stage'] = 'loading_profiles'
    profiles: dict[str, dict] = {}

    if not os.path.exists(jsonl_path):
        logger.warning('JSONL not found at %s — full profiles unavailable.', jsonl_path)
        return profiles

    logger.info('Scanning JSONL for %d candidates...', len(candidate_ids))
    t0 = time.time()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(profiles) == len(candidate_ids):
                break
            # Fast pre-check to avoid JSON parsing every line
            if not any(cid in line for cid in candidate_ids):
                continue
            try:
                cand = json.loads(line)
                if cand.get('candidate_id') in candidate_ids:
                    profiles[cand['candidate_id']] = cand
            except json.JSONDecodeError:
                continue

    logger.info('Loaded %d/%d profiles in %.1fs', len(profiles), len(candidate_ids), time.time() - t0)
    return profiles


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    jd_text: str,
    jsonl_path: str | None = None,
    top_k_retrieval: int = 500,
    shortlist_ratio: float = 0.25,
    critic_top_n: int = 30,
    output_top_n: int = 100,
    progress: dict | None = None,
    batch_id: str | None = None,
) -> dict:
    """
    Run the full 6-stage multi-agent pipeline.

    Args:
        jd_text:          Raw job description text
        jsonl_path:       Path to candidates.jsonl (for full profile extraction)
        top_k_retrieval:  How many candidates to pull from FAISS initially
        shortlist_ratio:  Fraction to shortlist for Stage 3 (default 25%)
        critic_top_n:     How many to pass through Stage 4 critic
        output_top_n:     Number of candidates in final output
        progress:         Mutable dict for stage progress updates (for streaming)

    Returns:
        {
          'results': [RankedCandidate.to_dict(), ...],
          'time_seconds': float,
          'stage_times': {stage: seconds},
          'pipeline_mode': 'agentic' | 'heuristic',
          'spec': RequirementSpec dict,
        }
    """
    pass
    if progress is None:
        progress = {}

    t_total = time.time()
    stage_times: dict[str, float] = {}

    base_dir   = Path(__file__).parent.parent
    _jsonl_path = jsonl_path or str(base_dir / 'dataset' / 'candidates.jsonl')

    # ------------------------------------------------------------------
    # Stage 1 — JD Decomposition
    # ------------------------------------------------------------------
    progress['stage'] = 'stage1_decomposing'
    logger.info('=== Stage 1: JD Decomposition ===')
    t1 = time.time()
    spec: RequirementSpec = decompose(jd_text)
    stage_times['stage1'] = round(time.time() - t1, 2)
    logger.info('Stage 1 done in %.2fs. must_have_skills=%s', stage_times['stage1'],
                spec.must_have_skills[:5])

    # ------------------------------------------------------------------
    # Stage 2 — Fast Pre-Filter (FAISS)
    # ------------------------------------------------------------------
    progress['stage'] = 'stage2_retrieving'
    logger.info('=== Stage 2: Supabase pgvector Pre-Filter ===')
    t2 = time.time()
    shortlisted = retrieve(
        spec,
        _faiss_index,
        _metadata_df,
        top_k=top_k_retrieval,
        shortlist_ratio=shortlist_ratio,
        batch_id=batch_id,
    )
    if not shortlisted:
        raise RuntimeError("No candidates found in your cloud database! Please use the 'Upload Candidates' button to add resumes first.")
    
    stage_times['stage2'] = round(time.time() - t2, 2)
    logger.info('Stage 2 done in %.2fs. Shortlist size: %d', stage_times['stage2'], len(shortlisted))

    # ------------------------------------------------------------------
    # Load full profiles for shortlisted candidates
    # ------------------------------------------------------------------
    candidate_ids = {c['candidate_id'] for c in shortlisted}
    full_profiles = _load_full_profiles(candidate_ids, _jsonl_path, progress)

    # Augment shortlist metadata with parquet row for Stage 3 fallback
    # (already in shortlisted[i]['metadata'])

    # ------------------------------------------------------------------
    # Stage 3 — Per-Candidate Evaluation (concurrent)
    # ------------------------------------------------------------------
    progress['stage'] = 'stage3_evaluating'
    logger.info('=== Stage 3: Per-Candidate Evaluation (%d candidates) ===', len(shortlisted))
    t3 = time.time()
    eval_results = evaluate_batch(shortlisted, full_profiles, spec)
    stage_times['stage3'] = round(time.time() - t3, 2)
    logger.info('Stage 3 done in %.2fs.', stage_times['stage3'])

    # ------------------------------------------------------------------
    # Stage 4 — Critic / Verifier
    # ------------------------------------------------------------------
    progress['stage'] = 'stage4_critiquing'
    logger.info('=== Stage 4: Critic/Verifier (top %d) ===', critic_top_n)
    t4 = time.time()
    critic_results = critique_batch(
        eval_results,
        full_profiles,
        spec.summary_for_embedding,
        top_n=critic_top_n,
    )
    stage_times['stage4'] = round(time.time() - t4, 2)
    logger.info('Stage 4 done in %.2fs.', stage_times['stage4'])

    # ------------------------------------------------------------------
    # Stage 5 — Synthesis & Ranking
    # ------------------------------------------------------------------
    progress['stage'] = 'stage5_synthesizing'
    logger.info('=== Stage 5: Output Synthesis ===')
    t5 = time.time()
    ranked = synthesize(critic_results, spec, top_n=output_top_n)
    stage_times['stage5'] = round(time.time() - t5, 2)
    logger.info('Stage 5 done in %.2fs.', stage_times['stage5'])

    total_time = round(time.time() - t_total, 2)
    progress['stage'] = 'done'

    # Determine if we used LLM anywhere
    llm_used = (
        any(r.used_llm for r in eval_results) or
        any(r.used_llm for r in critic_results)
    )

    logger.info(
        '=== Pipeline complete in %.2fs. LLM: %s. Top candidate: %s (%.4f) ===',
        total_time,
        llm_used,
        ranked[0].candidate_id if ranked else 'none',
        ranked[0].score if ranked else 0,
    )

    # Attach metadata from shortlist back to ranked results for frontend display
    shortlist_meta = {c['candidate_id']: c['metadata'] for c in shortlisted}
    result_dicts = []
    for r in ranked:
        d = r.to_dict()
        d['metadata'] = shortlist_meta.get(r.candidate_id, d['metadata'])
        result_dicts.append(d)

    return {
        'results':       result_dicts,
        'time_seconds':  total_time,
        'stage_times':   stage_times,
        'pipeline_mode': 'agentic' if llm_used else 'heuristic',
        'spec':          spec.to_dict(),
    }


# ---------------------------------------------------------------------------
# CLI entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description='Run the multi-agent ranking pipeline.')
    parser.add_argument('--jd',        type=str, help='Job description text or @file path')
    parser.add_argument('--candidates',type=str, default='dataset/candidates.jsonl')
    parser.add_argument('--index',     type=str, default='candidates.index')
    parser.add_argument('--metadata',  type=str, default='candidates_metadata.parquet')
    parser.add_argument('--out',       type=str, default='output_agentic.json')
    args = parser.parse_args()

    jd_text = args.jd or """
    Senior AI Engineer - Founding Team. 5-9 years of experience.
    Deep technical depth in modern ML systems - embeddings, retrieval, ranking, LLMs, fine-tuning.
    Production experience with embeddings-based retrieval systems.
    Production experience with vector databases - Pinecone, Weaviate, Qdrant, FAISS.
    Strong Python. Hands-on with NDCG, MRR ranking evaluation frameworks.
    Experience at product companies (not pure services/consulting).
    """
    if jd_text.startswith('@'):
        with open(jd_text[1:]) as f:
            jd_text = f.read()

    load_resources(args.index, args.metadata)
    result = run(jd_text, jsonl_path=args.candidates)

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Done. Saved to {args.out}. Time: {result['time_seconds']}s. "
          f"Mode: {result['pipeline_mode']}")
