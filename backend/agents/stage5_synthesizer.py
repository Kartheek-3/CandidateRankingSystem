"""
Stage 5 — Ranking & Output Synthesis
======================================
Aggregates critic-adjusted sub-scores into a final weighted rank, using
the inferred_weights from Stage 1's RequirementSpec — so the weights themselves
reflect what the JD actually emphasizes, not a hard-coded formula.

Formula:
  final_score = (
      weights.technical_skills  × adjusted_must_have_score
    + weights.experience_fit    × experience_score
    + weights.domain_fit        × domain_fit_score
    + weights.behavioral        × behavioral_score
  ) × semantic_boost

Where semantic_boost = 1.0 + (semantic_score - 0.5) × 0.1
(rewards candidates whose overall profile is closer to the JD embedding)

Outputs the final ranked list, each entry with:
  rank, candidate_id, score (0-1 normalized), reasoning, sub_scores, risk_flag
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

from .stage1_jd_decomposer import RequirementSpec
from .stage4_critic import CriticResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class RankedCandidate:
    rank:         int
    candidate_id: str
    score:        float          # final composite, rounded to 4dp
    reasoning:    str            # compact human-readable rationale
    sub_scores:   dict           # {technical, experience, domain, behavioral, semantic}
    risk_flag:    str | None
    metadata:     dict           # from parquet row (title, YOE, etc.)

    def to_dict(self) -> dict:
        return {
            'rank':         self.rank,
            'candidate_id': self.candidate_id,
            'score':        self.score,
            'reasoning':    self.reasoning,
            'sub_scores':   self.sub_scores,
            'risk_flag':    self.risk_flag,
            'metadata':     self.metadata,
        }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _compute_final_score(
    critic: CriticResult,
    weights: dict,
) -> float:
    ev = critic.eval_result

    must_have_score = critic.adjusted_must_have_score

    weighted = (
        weights.get('technical_skills', 0.45) * must_have_score
        + weights.get('experience_fit',   0.25) * ev.experience_score
        + weights.get('domain_fit',       0.20) * ev.domain_fit_score
        + weights.get('behavioral',       0.10) * ev.behavioral_score
    )

    # Semantic boost: small multiplicative bonus for candidates whose
    # full profile embedding is closer to the JD spec embedding.
    semantic_boost = 1.0 + (ev.semantic_score - 0.5) * 0.10
    final = weighted * semantic_boost

    # Nice-to-have skills add a small bonus (max +0.03)
    nice_bonus = min(0.03, ev.nice_to_have_score * 0.05)
    final += nice_bonus

    return round(min(max(final, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Rationale generation
# ---------------------------------------------------------------------------

def _build_reasoning(critic: CriticResult, metadata: dict) -> str:
    """Generate a compact, factual rationale from the evaluation data."""
    ev    = critic.eval_result
    title = metadata.get('current_title', 'Engineer')
    yoe   = metadata.get('years_of_experience', 0)

    parts = [f"{title} with {yoe:.0f} years of experience."]

    # Top claims from Stage 3
    if ev.verifiable_claims:
        top_claims = ev.verifiable_claims[:2]
        parts.append(' '.join(top_claims))

    # Must-have score context
    if ev.must_have_score >= 0.80:
        parts.append("Strong coverage of required technical skills.")
    elif ev.must_have_score >= 0.55:
        parts.append("Moderate coverage of required skills.")
    else:
        parts.append("Partial skill match — may require assessment.")

    # Behavioral signal
    if ev.behavioral_score >= 0.70:
        parts.append("Strong engagement signals.")
    elif ev.behavioral_score < 0.40:
        parts.append("Low platform engagement signals noted.")

    # Risk flag preview
    if critic.risk_flag:
        # Shorten if very long
        flag_preview = critic.risk_flag[:120] + ('...' if len(critic.risk_flag) > 120 else '')
        parts.append(f"⚠ Risk: {flag_preview}")

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def synthesize(
    critic_results: list[CriticResult],
    spec: RequirementSpec,
    top_n: int = 100,
) -> list[RankedCandidate]:
    """
    Stage 5 public interface.

    Args:
        critic_results: All CriticResult objects from Stage 4
        spec:           RequirementSpec (for inferred_weights)
        top_n:          How many candidates to include in final output

    Returns:
        Sorted list of RankedCandidate (rank 1 = best).
    """
    weights = spec.inferred_weights
    logger.info('[Stage5] Synthesizing %d candidates. Weights: %s', len(critic_results), weights)

    scored: list[tuple[float, str, CriticResult]] = []
    for critic in critic_results:
        final_score = _compute_final_score(critic, weights)
        scored.append((final_score, critic.candidate_id, critic))

    # Sort: descending score, then ascending candidate_id as tiebreaker
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:top_n]

    ranked: list[RankedCandidate] = []
    for rank_idx, (final_score, cid, critic) in enumerate(top, start=1):
        ev = critic.eval_result   # EvaluationResult directly

        # Minimal metadata dict (parquet metadata will be merged by orchestrator)
        parquet_meta: dict = {}

        sub_scores = {
            'technical':    round(critic.adjusted_must_have_score, 3),
            'experience':   round(ev.experience_score, 3),
            'domain':       round(ev.domain_fit_score, 3),
            'behavioral':   round(ev.behavioral_score, 3),
            'semantic':     round(ev.semantic_score, 3),
            'nice_to_have': round(ev.nice_to_have_score, 3),
        }

        reasoning = _build_reasoning(critic, parquet_meta)

        ranked.append(RankedCandidate(
            rank         = rank_idx,
            candidate_id = cid,
            score        = final_score,
            reasoning    = reasoning,
            sub_scores   = sub_scores,
            risk_flag    = critic.risk_flag,
            metadata     = parquet_meta,
        ))

    logger.info('[Stage5] Synthesis complete. Top candidate: %s (score=%.4f)',
                ranked[0].candidate_id if ranked else 'none',
                ranked[0].score if ranked else 0)
    return ranked
