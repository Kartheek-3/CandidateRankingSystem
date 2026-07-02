"""
rank.py — CLI Ranking Script
==============================
Modes:
  Default (fast, <5 min):  FAISS + heuristic scoring only. No LLM. Submission-safe.
  --agentic:               Full multi-agent pipeline (Stages 1-5). Requires GOOGLE_API_KEY.

Usage:
  python rank.py --candidates dataset/candidates.jsonl --out team_neural_runners.csv
  python rank.py --candidates dataset/candidates.jsonl --out output.csv --agentic
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('rank')


# ---------------------------------------------------------------------------
# Embedded JD (the challenge JD — no dataset file required)
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer - Founding Team. 5-9 years of experience.
Deep technical depth in modern ML systems - embeddings, retrieval, ranking, LLMs, fine-tuning.
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
Production experience with vector databases or hybrid search infrastructure - Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python. Code quality matters.
Hands-on experience designing evaluation frameworks for ranking systems - NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.
Experience at product companies (not pure services/consulting).
"""


# ---------------------------------------------------------------------------
# Fast heuristic path (original pipeline, preserved for submission compliance)
# ---------------------------------------------------------------------------

def run_heuristic(args) -> list[dict]:
    """Original FAISS + heuristic pipeline. Always fast. No LLM."""
    import torch
    import numpy as np
    import pandas as pd
    import faiss
    from sentence_transformers import SentenceTransformer

    logger.info('Running FAST heuristic pipeline (no LLM)...')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_embedding = model.encode([JD_TEXT], convert_to_numpy=True)
    faiss.normalize_L2(jd_embedding)

    try:
        index = faiss.read_index(args.index)
        df    = pd.read_parquet(args.metadata)
    except Exception as e:
        logger.error('Failed to load precomputed files: %s', e)
        sys.exit(1)

    distances, indices = index.search(jd_embedding, 500)

    top_candidates = []
    for rank_idx, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        row          = df.iloc[idx]
        sem_score    = float(distances[0][rank_idx])
        score        = sem_score

        if row['is_consulting_only']:            score -= 0.15
        skill_text = row['all_skills']
        if any(s in skill_text for s in ['pinecone', 'faiss', 'weaviate', 'qdrant']): score += 0.10
        if any(s in skill_text for s in ['ranking', 'retrieval', 'sentence-transformers']):  score += 0.10
        if any(s in skill_text for s in ['ndcg', 'mrr']):                              score += 0.05
        exp = row['years_of_experience']
        if exp < 4:   score -= (4 - exp) * 0.05
        elif exp > 15: score -= (exp - 15) * 0.02
        resp = row['recruiter_response_rate']
        score += (resp - 0.5) * 0.1

        top_candidates.append({
            'candidate_id': row['candidate_id'],
            'score':        score,
            'metadata':     row.to_dict(),
        })

    top_candidates.sort(key=lambda x: (-x['score'], x['candidate_id']))
    top_100_meta = top_candidates[:100]
    top_100_ids  = {c['candidate_id']: c for c in top_100_meta}

    # Load full profiles for reasoning
    top_100_full = {}
    if os.path.exists(args.candidates):
        with open(args.candidates, 'r', encoding='utf-8') as f:
            for line in f:
                if any(cid in line for cid in top_100_ids):
                    try:
                        cand = json.loads(line)
                        if cand['candidate_id'] in top_100_ids:
                            top_100_full[cand['candidate_id']] = cand
                            if len(top_100_full) == 100:
                                break
                    except json.JSONDecodeError:
                        continue

    results = []
    for rank, c_meta in enumerate(top_100_meta, start=1):
        cid       = c_meta['candidate_id']
        cand_full = top_100_full.get(cid, {})
        reasoning = _heuristic_reasoning(cand_full, c_meta['score'], c_meta['metadata'])
        results.append({
            'rank':         rank,
            'candidate_id': cid,
            'score':        round(c_meta['score'], 4),
            'reasoning':    reasoning,
        })
    return results


def _heuristic_reasoning(cand: dict, score: float, meta: dict) -> str:
    prof         = cand.get('profile', {})
    sig          = cand.get('redrob_signals', {})
    exp          = prof.get('years_of_experience', meta.get('years_of_experience', 0))
    title        = prof.get('current_title', meta.get('current_title', 'Engineer'))
    skills_list  = [s.get('name', '').lower() for s in cand.get('skills', [])]
    target       = ['pinecone', 'faiss', 'weaviate', 'qdrant', 'sentence-transformers',
                    'embeddings', 'retrieval', 'ranking', 'python', 'ndcg', 'mrr']
    matched      = [s.title() for s in skills_list if any(t in s for t in target)][:3]

    parts = [f"{title} with {exp} years of experience."]
    if matched:
        parts.append(f"Demonstrates expertise in {', '.join(matched)}.")
    if meta.get('is_consulting_only'):
        parts.append("Primarily consulting experience — slight product-focus mismatch.")
    elif exp < 5:
        parts.append("Slightly below 5-year ideal band, but strong relevance.")
    elif exp > 9:
        parts.append("Highly experienced with deep domain knowledge.")
    resp = sig.get('recruiter_response_rate', meta.get('recruiter_response_rate', 0.5))
    if resp > 0.8:
        parts.append(f"Strong engagement signals ({resp:.0%} response rate).")
    elif resp < 0.3:
        parts.append(f"Low recruiter response rate ({resp:.0%}) — behavioral risk.")
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Agentic path
# ---------------------------------------------------------------------------

def run_agentic(args) -> list[dict]:
    """Full multi-agent pipeline — Stages 1-5."""
    from dotenv import load_dotenv
    load_dotenv()

    sys.path.insert(0, str(Path(__file__).parent / 'backend'))
    import orchestrator

    logger.info('Running AGENTIC multi-agent pipeline...')
    orchestrator.load_resources(args.index, args.metadata)
    result = orchestrator.run(JD_TEXT, jsonl_path=args.candidates)

    output = []
    for r in result['results']:
        sub = r.get('sub_scores', {})
        reasoning = r.get('reasoning', '')
        if r.get('risk_flag'):
            reasoning += f" [Risk: {r['risk_flag'][:80]}]"
        output.append({
            'rank':         r['rank'],
            'candidate_id': r['candidate_id'],
            'score':        r['score'],
            'reasoning':    reasoning,
        })
    logger.info('Pipeline mode: %s | Total time: %.2fs', result['pipeline_mode'], result['time_seconds'])
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Rank candidates against the JD.')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--out',        required=True, help='Output CSV path')
    parser.add_argument('--index',      default='candidates.index')
    parser.add_argument('--metadata',   default='candidates_metadata.parquet')
    parser.add_argument('--agentic',    action='store_true',
                        help='Use full multi-agent LLM pipeline (requires GOOGLE_API_KEY)')
    args = parser.parse_args()

    t0 = time.time()

    if args.agentic:
        results = run_agentic(args)
    else:
        results = run_heuristic(args)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for r in results:
            writer.writerow([r['candidate_id'], r['rank'], r['score'], r['reasoning']])

    elapsed = time.time() - t0
    logger.info('Done in %.1fs. Output: %s (%d candidates)', elapsed, args.out, len(results))


if __name__ == '__main__':
    main()
