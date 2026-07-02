"""
Stage 4 — Critic / Verifier Agent  (Self-Correction Loop)
============================================================
This is the genuinely agentic stage. It takes the top-N evaluation results from
Stage 3 and questions each score:

  - Is the must_have_score driven by genuine semantic fit or keyword coincidence?
  - Does the raw_rationale actually support the numeric scores?
  - Are there red flags (consulting-only, short tenure, low behavioral signals)
    that Stage 3 may have underweighted?

The critic CAN adjust scores down (by up to 0.15) or add a risk flag.
It does NOT inflate scores — only deflation + flagging.

LLM path:  Gemini 1.5 Flash — given both the profile AND Stage 3 output,
           asks "does the evidence support this score?"
Fallback:  Rule-based auditor that checks obvious inconsistencies.

Operates only on top-N from Stage 3 (default: top 30) to control cost/latency.
"""

from __future__ import annotations
import json
import os
import logging

from .stage3_evaluator import EvaluationResult

logger = logging.getLogger(__name__)

CRITIC_TOP_N = 30         # Only run critic on top-N candidates
MAX_ADJUSTMENT = 0.15     # Maximum downward score adjustment


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class CriticResult:
    def __init__(
        self,
        eval_result: EvaluationResult,
        score_adjustment: float = 0.0,
        risk_flag: str | None = None,
        critic_rationale: str = '',
        used_llm: bool = False,
    ):
        self.candidate_id    = eval_result.candidate_id
        self.eval_result     = eval_result
        # Adjusted scores (critic can only decrease must_have_score)
        self.adjusted_must_have_score = max(
            0.0, eval_result.must_have_score - abs(score_adjustment)
        )
        self.score_adjustment   = score_adjustment
        self.risk_flag          = risk_flag
        self.critic_rationale   = critic_rationale
        self.used_llm           = used_llm

    def to_dict(self) -> dict:
        base = self.eval_result.to_dict()
        base.update({
            'adjusted_must_have_score': round(self.adjusted_must_have_score, 3),
            'score_adjustment':         round(self.score_adjustment, 3),
            'risk_flag':                self.risk_flag,
            'critic_rationale':         self.critic_rationale,
            'critic_used_llm':          self.used_llm,
        })
        return base


# ---------------------------------------------------------------------------
# Rule-based auditor (fallback)
# ---------------------------------------------------------------------------

def _rule_based_audit(eval_result: EvaluationResult, raw: dict) -> CriticResult:
    """
    Heuristic critic — flags obvious inconsistencies without LLM.
    """
    adjustment  = 0.0
    risk_flag   = None
    rationale   = 'Rule-based audit passed.'

    must_score = eval_result.must_have_score
    sem_score  = eval_result.semantic_score
    skills     = raw.get('skills', [])
    expert_skills = [s['name'].lower() for s in skills if s.get('proficiency') == 'expert']
    adv_skills    = [s['name'].lower() for s in skills if s.get('proficiency') == 'advanced']
    all_skill_names = expert_skills + adv_skills

    # Rule 1: High must_have_score but low semantic_score → possible keyword inflation
    if must_score > 0.75 and sem_score < 0.55:
        adjustment = -0.10
        risk_flag  = 'High must-have score conflicts with low semantic similarity — possible keyword listing without depth.'
        rationale  = 'Must-have score adjusted down: strong skill claims not reflected in semantic profile embedding.'

    # Rule 2: Expert proficiency skills with very short duration
    suspicious_experts = [
        s['name'] for s in skills
        if s.get('proficiency') == 'expert' and 0 < s.get('duration_months', 99) < 6
    ]
    if suspicious_experts:
        adjustment = min(adjustment - 0.08, -0.08)
        risk = f"Expert-level claims with <6 months usage: {', '.join(suspicious_experts[:3])}"
        risk_flag = risk_flag + ' | ' + risk if risk_flag else risk
        rationale += f' Suspicious expert claims detected: {suspicious_experts[:3]}.'

    # Rule 3: Consulting-only but domain_fit_score is high
    profile_raw = raw.get('profile', {})
    career = raw.get('career_history', [])
    is_consulting = all(
        any(firm in str(j.get('company', '')).lower()
            for firm in ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'])
        for j in career
    )
    if is_consulting and eval_result.domain_fit_score > 0.75:
        adjustment = min(adjustment - 0.05, adjustment - 0.05)
        risk_note  = 'Domain fit score may be inflated — all roles are at consulting firms, not product companies.'
        risk_flag  = risk_flag + ' | ' + risk_note if risk_flag else risk_note

    # Rule 4: No advanced/expert skills at all but high must_have_score
    if not all_skill_names and must_score > 0.5:
        adjustment = -0.12
        risk_flag  = 'Must-have score suspicious — no advanced/expert skills found in profile.'
        rationale  = 'Score adjusted: skills listed without proficiency depth.'

    # Clamp adjustment
    adjustment = max(adjustment, -MAX_ADJUSTMENT)

    return CriticResult(
        eval_result     = eval_result,
        score_adjustment= adjustment,
        risk_flag       = risk_flag,
        critic_rationale= rationale,
        used_llm        = False,
    )


# ---------------------------------------------------------------------------
# LLM critic
# ---------------------------------------------------------------------------

_CRITIC_PROMPT = """You are a skeptical senior recruiter reviewing an AI-generated candidate evaluation.

REQUIREMENT SPEC:
{spec_summary}

CANDIDATE EVALUATION (from Stage 3):
- Candidate: {title}, {yoe} years experience
- Must-Have Score: {must_score:.2f}/1.0
- Domain Fit Score: {domain_score:.2f}/1.0
- Stage 3 Rationale: "{rationale}"
- Verifiable Claims: {claims}
- Key Skills (advanced/expert): {skills}
- Has Product Experience: {has_product}
- Semantic Similarity to JD: {sem_score:.3f}

Your task: Question whether the scores are justified by the EVIDENCE.

Respond ONLY with valid JSON — no markdown:
{{
  "score_adjustment": <float between -0.15 and 0.0, negative means reduce>,
  "risk_flag": "<null or a specific 1-sentence risk to flag>",
  "critic_rationale": "<1-2 sentences explaining your assessment>"
}}

Rules:
- Only adjust negatively (maximum -0.15). Never inflate scores.
- Set risk_flag to null if there are no concerns.
- Be specific — mention actual evidence or lack thereof."""


def _llm_critique(
    eval_result: EvaluationResult,
    raw: dict,
    spec_summary: str,
) -> CriticResult | None:
    """LLM critic call. Returns None on any failure."""
    api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        return None

    profile = raw.get('profile', {})
    skills  = raw.get('skills', [])
    expert_skills = ', '.join([
        s['name'] for s in skills
        if s.get('proficiency') in ['advanced', 'expert']
    ][:10])

    prompt = _CRITIC_PROMPT.format(
        spec_summary = spec_summary[:300],
        title        = profile.get('current_title', ''),
        yoe          = profile.get('years_of_experience', 0),
        must_score   = eval_result.must_have_score,
        domain_score = eval_result.domain_fit_score,
        rationale    = eval_result.raw_rationale[:200],
        claims       = eval_result.verifiable_claims[:3],
        skills       = expert_skills,
        has_product  = not bool(raw.get('_is_consulting_only', False)),
        sem_score    = eval_result.semantic_score,
    )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {"role": "system", "content": "You are a candidate verifier. Output JSON matching the schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content.strip())

        adjustment = float(data.get('score_adjustment', 0.0))
        adjustment = max(-MAX_ADJUSTMENT, min(0.0, adjustment))  # clamp

        return CriticResult(
            eval_result      = eval_result,
            score_adjustment = adjustment,
            risk_flag        = data.get('risk_flag') or None,
            critic_rationale = data.get('critic_rationale', ''),
            used_llm         = True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning('[Stage4] LLM critique failed for %s: %s', eval_result.candidate_id, exc)
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def critique_batch(
    eval_results: list[EvaluationResult],
    full_profiles: dict[str, dict],
    spec_summary: str,
    top_n: int = CRITIC_TOP_N,
) -> list[CriticResult]:
    """
    Run Stage 4 critic over the top-N evaluation results.

    Args:
        eval_results:  All Stage 3 results (sorted by composite score desc)
        full_profiles: Raw candidate dicts keyed by candidate_id
        spec_summary:  The RequirementSpec summary_for_embedding string
        top_n:         How many to run the critic over (rest pass through unchanged)

    Returns:
        Full list of CriticResult — top_n critiqued, rest wrapped as-is.
    """
    # Sort by must_have_score descending to focus critic on the top candidates
    sorted_results = sorted(eval_results, key=lambda r: -r.must_have_score)
    to_critique    = sorted_results[:top_n]
    pass_through   = sorted_results[top_n:]

    api_key = os.getenv('GOOGLE_API_KEY', '')
    logger.info('[Stage4] Critiquing top %d candidates (llm=%s).', len(to_critique), bool(api_key))

    critic_results: list[CriticResult] = []

    for eval_result in to_critique:
        raw = full_profiles.get(eval_result.candidate_id, {})
        result = _llm_critique(eval_result, raw, spec_summary) if api_key \
                 else _rule_based_audit(eval_result, raw)
        if result is None:
            result = _rule_based_audit(eval_result, raw)
        critic_results.append(result)

    # Wrap pass-through candidates (no adjustment, no risk flag)
    for eval_result in pass_through:
        critic_results.append(CriticResult(
            eval_result      = eval_result,
            score_adjustment = 0.0,
            risk_flag        = None,
            critic_rationale = 'Below critic threshold — passed through.',
            used_llm         = False,
        ))

    logger.info('[Stage4] Critique complete. %d results.', len(critic_results))
    return critic_results
