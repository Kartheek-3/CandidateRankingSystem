"""
Stage 3 — Per-Candidate Evaluation Agent
==========================================
For each shortlisted candidate, runs a structured multi-step evaluation:
  (a) Extract verifiable claims from the profile
  (b) Score must-haves and nice-to-haves separately against the RequirementSpec
  (c) Pull behavioral/activity signals as a distinct factor

LLM path:  Gemini 1.5 Flash — 3-step chain prompt
Fallback:  Heuristic rule-based scorer (keyword matching + signal math)

Output per candidate:
{
  "candidate_id":       "CAND_0001234",
  "must_have_score":    0.82,   # 0–1
  "nice_to_have_score": 0.55,   # 0–1
  "experience_score":   0.90,   # 0–1
  "behavioral_score":   0.71,   # 0–1
  "domain_fit_score":   0.75,   # 0–1 (inferred from career context)
  "verifiable_claims":  ["Used FAISS in prod at Meesho 18 months"],
  "raw_rationale":      "Strong match on vector DBs..."
}

Parallelism: run_batch() uses ThreadPoolExecutor — runs concurrently across candidates.
"""

from __future__ import annotations
import json
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from .stage0_ingestion import CandidateProfile, CandidateActivity, CandidateSkills
from .stage1_jd_decomposer import RequirementSpec

logger = logging.getLogger(__name__)

MAX_WORKERS = 8   # concurrent LLM calls
LLM_TIMEOUT = 20  # seconds per call


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    candidate_id:       str
    semantic_score:     float   # from Stage 2
    must_have_score:    float
    nice_to_have_score: float
    experience_score:   float
    behavioral_score:   float
    domain_fit_score:   float
    verifiable_claims:  list[str]
    raw_rationale:      str
    used_llm:           bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Heuristic fallback evaluator
# ---------------------------------------------------------------------------

def _score_must_haves_heuristic(
    skills_text: str,
    skill_list: list[dict],
    spec: RequirementSpec,
) -> float:
    matched = sum(1 for skill in spec.must_have_skills if skill.lower() in skills_text)
    total   = max(len(spec.must_have_skills), 1)
    return round(min(matched / total, 1.0), 3)


def _score_nice_to_have_heuristic(skills_text: str, spec: RequirementSpec) -> float:
    matched = sum(1 for skill in spec.nice_to_have_skills if skill.lower() in skills_text)
    total   = max(len(spec.nice_to_have_skills), 1)
    return round(min(matched / total, 1.0), 3)


def _score_experience_heuristic(yoe: float, spec: RequirementSpec) -> float:
    lo, hi = spec.experience_band.get('min', 3), spec.experience_band.get('max', 10)
    if lo <= yoe <= hi:
        return 1.0
    if yoe < lo:
        deficit = lo - yoe
        return max(0.0, round(1.0 - deficit * 0.08, 3))
    # above band — slight penalty for over-qualification
    excess = yoe - hi
    return max(0.5, round(1.0 - excess * 0.03, 3))


def _score_behavioral_heuristic(activity: CandidateActivity) -> float:
    score = 0.5  # base
    score += (activity.recruiter_response_rate - 0.5) * 0.15
    score += (activity.interview_completion_rate - 0.5) * 0.10
    if activity.github_activity_score >= 0:
        score += (activity.github_activity_score / 100 - 0.5) * 0.08
    if activity.open_to_work:
        score += 0.05
    if activity.verified_email and activity.verified_phone:
        score += 0.03
    profile_bonus = (activity.profile_completeness_score / 100 - 0.5) * 0.06
    score += profile_bonus
    return round(min(max(score, 0.0), 1.0), 3)


def _score_domain_fit_heuristic(
    profile: CandidateProfile,
    spec: RequirementSpec,
) -> float:
    score = 0.5
    if spec.product_company_required and profile.has_product_experience:
        score += 0.30
    elif spec.product_company_required and profile.is_consulting_only:
        score -= 0.20
    # Industry alignment
    industry = profile.current_industry.lower()
    domain   = spec.domain_context.lower()
    overlap_words = set(industry.split()) & set(domain.split())
    score += len(overlap_words) * 0.05
    return round(min(max(score, 0.0), 1.0), 3)


def _heuristic_evaluate(
    profile: CandidateProfile,
    activity: CandidateActivity,
    skills: CandidateSkills,
    spec: RequirementSpec,
    semantic_score: float,
) -> EvaluationResult:
    """Full heuristic evaluation — no LLM required."""
    must_have_score    = _score_must_haves_heuristic(skills.all_skills_text, skills.skills, spec)
    nice_to_have_score = _score_nice_to_have_heuristic(skills.all_skills_text, spec)
    experience_score   = _score_experience_heuristic(profile.years_of_experience, spec)
    behavioral_score   = _score_behavioral_heuristic(activity)
    domain_fit_score   = _score_domain_fit_heuristic(profile, spec)

    # Build basic rationale
    matched_musts = [s for s in spec.must_have_skills if s.lower() in skills.all_skills_text]
    claims = [
        f"{profile.current_title} with {profile.years_of_experience:.0f} years experience",
    ]
    if matched_musts:
        claims.append(f"Skills matched: {', '.join(matched_musts[:4])}")
    if profile.has_product_experience:
        claims.append('Has product company experience')

    rationale = (
        f"{profile.current_title}, {profile.years_of_experience:.0f} YOE. "
        f"Must-have coverage: {must_have_score:.0%}. "
        f"{'Product company exp.' if profile.has_product_experience else 'Consulting background.'}"
    )

    return EvaluationResult(
        candidate_id       = profile.candidate_id,
        semantic_score     = semantic_score,
        must_have_score    = must_have_score,
        nice_to_have_score = nice_to_have_score,
        experience_score   = experience_score,
        behavioral_score   = behavioral_score,
        domain_fit_score   = domain_fit_score,
        verifiable_claims  = claims,
        raw_rationale      = rationale,
        used_llm           = False,
    )


# ---------------------------------------------------------------------------
# LLM evaluator
# ---------------------------------------------------------------------------

_EVAL_PROMPT_TEMPLATE = """You are a technical recruiter AI. Evaluate the following candidate against a job requirement spec.

REQUIREMENT SPEC:
{spec_json}

CANDIDATE PROFILE:
Title: {title}
Experience: {yoe} years
Career Summary: {summary}
Skills (advanced/expert): {skills}
Career History (last 3 roles): {career}
Product Company Experience: {has_product}

Respond ONLY with valid JSON matching this exact schema — no markdown, no extra text:
{{
  "verifiable_claims": ["<specific factual claim from profile>", ...],
  "must_have_score": <0.0-1.0>,
  "nice_to_have_score": <0.0-1.0>,
  "experience_score": <0.0-1.0>,
  "domain_fit_score": <0.0-1.0>,
  "raw_rationale": "<2-3 sentence explanation referencing specific evidence>"
}}

Scoring guidance:
- must_have_score: fraction of must-have skills genuinely evidenced (not just listed)
- experience_score: 1.0 if YOE in spec band, penalise proportionally outside
- domain_fit_score: does career context match the domain? product experience if required?
- Be conservative — only score high if evidence is clear, not just claimed."""


def _llm_evaluate_single(
    profile: CandidateProfile,
    activity: CandidateActivity,
    skills: CandidateSkills,
    spec: RequirementSpec,
    semantic_score: float,
) -> EvaluationResult | None:
    """Single-candidate LLM evaluation. Returns None on failure."""
    api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        return None

    # Build career excerpt (last 3 roles only to save tokens)
    career_excerpt = ' | '.join([
        f"{j.get('title')} @ {j.get('company')} ({j.get('duration_months')}mo)"
        for j in profile.career_history[:3]
    ])
    expert_skills = ', '.join([
        s['name'] for s in skills.skills
        if s.get('proficiency') in ['advanced', 'expert']
    ][:12])

    prompt = _EVAL_PROMPT_TEMPLATE.format(
        spec_json   = json.dumps(spec.to_dict(), indent=2)[:800],
        title       = profile.current_title,
        yoe         = profile.years_of_experience,
        summary     = profile.summary[:300],
        skills      = expert_skills,
        career      = career_excerpt,
        has_product = profile.has_product_experience,
    )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {"role": "system", "content": "You are a candidate evaluator. Output JSON matching the schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)

        behavioral_score = _score_behavioral_heuristic(activity)

        return EvaluationResult(
            candidate_id       = profile.candidate_id,
            semantic_score     = semantic_score,
            must_have_score    = float(data.get('must_have_score', 0.5)),
            nice_to_have_score = float(data.get('nice_to_have_score', 0.3)),
            experience_score   = float(data.get('experience_score', 0.5)),
            behavioral_score   = behavioral_score,
            domain_fit_score   = float(data.get('domain_fit_score', 0.5)),
            verifiable_claims  = data.get('verifiable_claims', []),
            raw_rationale      = data.get('raw_rationale', ''),
            used_llm           = True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning('[Stage3] LLM eval failed for %s: %s', profile.candidate_id, exc)
        return None


# ---------------------------------------------------------------------------
# Candidate entity assembly (from raw full JSON)
# ---------------------------------------------------------------------------

def _build_entities(
    candidate_raw: dict,
    meta_row: dict,
    semantic_score: float,
) -> tuple:
    """Build typed entities from raw dict + metadata row."""
    from .stage0_ingestion import normalize, CandidateProfile, CandidateActivity, CandidateSkills

    result = normalize(candidate_raw)
    if result is None:
        # Fallback — build minimal entities from metadata
        profile = CandidateProfile(
            candidate_id          = meta_row.get('candidate_id', ''),
            anonymized_name       = '',
            headline              = '',
            summary               = '',
            location              = '',
            country               = '',
            years_of_experience   = float(meta_row.get('profile', {}).get('years_of_experience', meta_row.get('years_of_experience', 0))),
            current_title         = meta_row.get('profile', {}).get('current_title', meta_row.get('current_title', '')),
            current_company       = '',
            current_company_size  = '',
            current_industry      = '',
            career_history        = [],
            education             = [],
            certifications        = [],
            is_consulting_only    = bool(meta_row.get('is_consulting_only', False)),
            has_product_experience= not bool(meta_row.get('is_consulting_only', False)),
        )
        activity = CandidateActivity(
            candidate_id              = meta_row.get('candidate_id', ''),
            profile_completeness_score= float(meta_row.get('profile_completeness_score', 0)),
            last_active_date          = meta_row.get('last_active_date', ''),
            open_to_work              = False,
            profile_views_30d         = 0,
            applications_30d          = 0,
            recruiter_response_rate   = float(meta_row.get('recruiter_response_rate', 0.5)),
            avg_response_time_hours   = 24.0,
            interview_completion_rate = float(meta_row.get('interview_completion_rate', 0.5)),
            offer_acceptance_rate     = -1,
            github_activity_score     = -1,
            saved_by_recruiters_30d   = 0,
            skill_assessment_scores   = {},
            verified_email            = False,
            verified_phone            = False,
            linkedin_connected        = False,
            notice_period_days        = 30,
            willing_to_relocate       = False,
            preferred_work_mode       = 'flexible',
        )
        all_skills_text = meta_row.get('all_skills', '')
        skill_entities = CandidateSkills(
            candidate_id    = meta_row.get('candidate_id', ''),
            skills          = [],
            all_skills_text = all_skills_text,
        )
        return profile, activity, skill_entities
    return result


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate_single(
    candidate_raw: dict,
    meta_row: dict,
    semantic_score: float,
    spec: RequirementSpec,
) -> EvaluationResult:
    """
    Evaluate one candidate. Tries LLM first, falls back to heuristic.
    """
    profile, activity, skills = _build_entities(candidate_raw, meta_row, semantic_score)
    result = _llm_evaluate_single(profile, activity, skills, spec, semantic_score)
    if result is None:
        result = _heuristic_evaluate(profile, activity, skills, spec, semantic_score)
    return result


def run_batch(
    shortlisted: list[dict],
    full_profiles: dict[str, dict],
    spec: RequirementSpec,
) -> list[EvaluationResult]:
    """
    Run Stage 3 evaluation concurrently over all shortlisted candidates.

    Args:
        shortlisted:   list of {candidate_id, semantic_score, metadata} from Stage 2
        full_profiles: dict of candidate_id → full raw dict from JSONL
        spec:          RequirementSpec from Stage 1

    Returns:
        List of EvaluationResult objects (unordered — Stage 5 handles ranking).
    """
    results: list[EvaluationResult] = []

    def _worker(item: dict) -> EvaluationResult:
        cid          = item['candidate_id']
        sem_score    = item['semantic_score']
        meta_row     = item['metadata']
        raw_profile  = full_profiles.get(cid, {})
        return evaluate_single(raw_profile, meta_row, sem_score, spec)

    api_key = os.getenv('GOOGLE_API_KEY', '')
    n_workers = MAX_WORKERS if api_key else 1

    logger.info('[Stage3] Evaluating %d candidates (workers=%d, llm=%s)',
                len(shortlisted), n_workers, bool(api_key))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, item): item for item in shortlisted}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.error('[Stage3] Worker failed: %s', exc)

    logger.info('[Stage3] Evaluation complete. %d results.', len(results))
    return results
