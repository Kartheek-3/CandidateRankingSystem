"""
Stage 0 — Ingestion & Normalization
====================================
Pure-Python plumbing. Takes a raw candidate dict (from JSONL) and splits it
into three clean, typed entities:
  - CandidateProfile   (static / demographic attributes)
  - CandidateActivity  (behavioral / Redrob-signals)
  - CandidateSkills    (structured skill list)

This separation means every downstream agent receives only the data it needs,
and missing-value handling happens exactly once, right here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Output entity dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateProfile:
    candidate_id: str
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    career_history: list[dict]
    education: list[dict]
    certifications: list[dict]
    is_consulting_only: bool          # derived flag
    has_product_experience: bool      # derived flag


@dataclass
class CandidateActivity:
    candidate_id: str
    profile_completeness_score: float
    last_active_date: str
    open_to_work: bool
    profile_views_30d: int
    applications_30d: int
    recruiter_response_rate: float    # 0–1
    avg_response_time_hours: float
    interview_completion_rate: float  # 0–1
    offer_acceptance_rate: float      # 0–1 or -1 if no history
    github_activity_score: float      # 0–100 or -1 if not linked
    saved_by_recruiters_30d: int
    skill_assessment_scores: dict[str, float]
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
    notice_period_days: int
    willing_to_relocate: bool
    preferred_work_mode: str


@dataclass
class CandidateSkills:
    candidate_id: str
    skills: list[dict]                # [{name, proficiency, endorsements, duration_months}]
    all_skills_text: str              # flat lowercase join for fast substring scan


# ---------------------------------------------------------------------------
# Consulting-firm heuristic (same logic as precompute.py, centralised here)
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'hcl', 'mphasis', 'l&t infotech', 'ltimindtree', 'tech mahindra',
}


def _detect_company_type(career_history: list[dict]) -> tuple[bool, bool]:
    """Return (is_consulting_only, has_product_experience)."""
    has_product = False
    is_consulting_only = True
    for job in career_history:
        company = str(job.get('company', '')).lower().strip()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            has_product = True
            is_consulting_only = False
    return is_consulting_only, has_product


# ---------------------------------------------------------------------------
# Honeypot detection — identical to precompute.py, kept here for completeness
# ---------------------------------------------------------------------------

def is_honeypot(raw: dict) -> bool:
    """Expert skill with 0 duration is a known honeypot signal."""
    for skill in raw.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------

def normalize(raw: dict) -> tuple[CandidateProfile, CandidateActivity, CandidateSkills] | None:
    """
    Parse a raw candidate dict into the three entity types.
    Returns None if the record should be discarded (honeypot, missing ID).
    """
    cid = raw.get('candidate_id', '')
    if not cid:
        return None
    if is_honeypot(raw):
        return None

    prof_raw = raw.get('profile', {})
    sig_raw  = raw.get('redrob_signals', {})
    career   = raw.get('career_history', [])
    skills   = raw.get('skills', [])

    is_consulting, has_product = _detect_company_type(career)

    profile = CandidateProfile(
        candidate_id          = cid,
        anonymized_name       = prof_raw.get('anonymized_name', ''),
        headline              = prof_raw.get('headline', ''),
        summary               = prof_raw.get('summary', ''),
        location              = prof_raw.get('location', ''),
        country               = prof_raw.get('country', ''),
        years_of_experience   = float(prof_raw.get('years_of_experience', 0)),
        current_title         = prof_raw.get('current_title', ''),
        current_company       = prof_raw.get('current_company', ''),
        current_company_size  = prof_raw.get('current_company_size', ''),
        current_industry      = prof_raw.get('current_industry', ''),
        career_history        = career,
        education             = raw.get('education', []),
        certifications        = raw.get('certifications', []),
        is_consulting_only    = is_consulting,
        has_product_experience= has_product,
    )

    activity = CandidateActivity(
        candidate_id              = cid,
        profile_completeness_score= float(sig_raw.get('profile_completeness_score', 0)),
        last_active_date          = sig_raw.get('last_active_date', ''),
        open_to_work              = bool(sig_raw.get('open_to_work_flag', False)),
        profile_views_30d         = int(sig_raw.get('profile_views_received_30d', 0)),
        applications_30d          = int(sig_raw.get('applications_submitted_30d', 0)),
        recruiter_response_rate   = float(sig_raw.get('recruiter_response_rate', 0.5)),
        avg_response_time_hours   = float(sig_raw.get('avg_response_time_hours', 24)),
        interview_completion_rate = float(sig_raw.get('interview_completion_rate', 0.5)),
        offer_acceptance_rate     = float(sig_raw.get('offer_acceptance_rate', -1)),
        github_activity_score     = float(sig_raw.get('github_activity_score', -1)),
        saved_by_recruiters_30d   = int(sig_raw.get('saved_by_recruiters_30d', 0)),
        skill_assessment_scores   = sig_raw.get('skill_assessment_scores', {}),
        verified_email            = bool(sig_raw.get('verified_email', False)),
        verified_phone            = bool(sig_raw.get('verified_phone', False)),
        linkedin_connected        = bool(sig_raw.get('linkedin_connected', False)),
        notice_period_days        = int(sig_raw.get('notice_period_days', 30)),
        willing_to_relocate       = bool(sig_raw.get('willing_to_relocate', False)),
        preferred_work_mode       = sig_raw.get('preferred_work_mode', 'flexible'),
    )

    all_skills_text = ' '.join(s.get('name', '').lower() for s in skills)
    skill_entities = CandidateSkills(
        candidate_id    = cid,
        skills          = skills,
        all_skills_text = all_skills_text,
    )

    return profile, activity, skill_entities
