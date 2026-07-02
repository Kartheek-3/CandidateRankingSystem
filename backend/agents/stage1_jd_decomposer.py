"""
Stage 1 — JD Decomposition Agent
==================================
Takes raw Job Description text and produces a structured RequirementSpec.
This spec is the *single rubric* all later stages score against.

LLM path:   Gemini 1.5 Flash → JSON structured output
Fallback:   Regex + keyword heuristic parser (no API key required)

Output RequirementSpec example:
{
  "must_have_skills":   ["faiss", "python", "embeddings", "vector databases"],
  "nice_to_have_skills":["ndcg", "mrr", "fine-tuning", "llms"],
  "experience_band":    {"min": 5, "max": 9},
  "domain_context":     "AI/ML infrastructure at a product-focused company",
  "seniority_label":    "senior",
  "product_company_required": true,
  "inferred_weights":   {
      "technical_skills": 0.45,
      "experience_fit":   0.25,
      "domain_fit":       0.20,
      "behavioral":       0.10
  },
  "summary_for_embedding": "Senior AI engineer 5-9 years embeddings retrieval ranking ..."
}
"""

from __future__ import annotations
import json
import os
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RequirementSpec — the rubric object passed to all downstream stages
# ---------------------------------------------------------------------------

class RequirementSpec:
    def __init__(self, data: dict):
        self.must_have_skills:         list[str]       = data.get('must_have_skills', [])
        self.nice_to_have_skills:      list[str]       = data.get('nice_to_have_skills', [])
        self.experience_band:          dict            = data.get('experience_band', {'min': 0, 'max': 99})
        self.domain_context:           str             = data.get('domain_context', '')
        self.seniority_label:          str             = data.get('seniority_label', 'mid')
        self.product_company_required: bool            = data.get('product_company_required', False)
        self.inferred_weights:         dict            = data.get('inferred_weights', {
            'technical_skills': 0.45,
            'experience_fit':   0.25,
            'domain_fit':       0.20,
            'behavioral':       0.10,
        })
        self.summary_for_embedding: str = data.get('summary_for_embedding', '')
        self._raw = data

    def to_dict(self) -> dict:
        return self._raw


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a recruiting analyst. Given a job description, extract a structured requirement spec.
Return ONLY valid JSON, no markdown fences, no explanation. Use this exact schema:
{
  "must_have_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill3"],
  "experience_band": {"min": <int>, "max": <int>},
  "domain_context": "<one sentence>",
  "seniority_label": "<junior|mid|senior|staff|principal>",
  "product_company_required": <true|false>,
  "inferred_weights": {
    "technical_skills": <0.0-1.0>,
    "experience_fit": <0.0-1.0>,
    "domain_fit": <0.0-1.0>,
    "behavioral": <0.0-1.0>
  },
  "summary_for_embedding": "<compact 30-word summary of ideal candidate>"
}
Weights must sum to 1.0. Be precise about must_have vs nice_to_have — only put skills explicitly emphasized as required/must in must_have_skills."""


# ---------------------------------------------------------------------------
# Heuristic fallback parser
# ---------------------------------------------------------------------------

# Common AI/ML skill taxonomy
_MUST_HAVE_KEYWORDS = [
    'python', 'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
    'opensearch', 'elasticsearch', 'embeddings', 'sentence-transformers',
    'retrieval', 'ranking', 'vector database', 'vector db', 'llm', 'fine-tuning',
    'machine learning', 'deep learning', 'pytorch', 'tensorflow',
]
_NICE_TO_HAVE_KEYWORDS = [
    'ndcg', 'mrr', 'map', 'a/b test', 'spark', 'kafka', 'kubernetes',
    'docker', 'aws', 'gcp', 'azure', 'mlflow', 'ray', 'triton',
    'openai', 'langchain', 'llama', 'bge', 'e5',
]
_EXP_PATTERN = re.compile(
    r'(\d+)\s*[-–to]+\s*(\d+)\s*years?|(\d+)\+\s*years?|(\d+)\s*years?\s*(?:of\s*)?experience',
    re.IGNORECASE
)
_SENIORITY_MAP = {
    'junior': ['junior', 'associate', 'entry'],
    'mid': ['mid', 'intermediate'],
    'senior': [r'senior', r'sr\.', r'lead'],
    'staff': ['staff'],
    'principal': ['principal', 'architect', 'distinguished'],
}


def _heuristic_parse(jd_text: str) -> RequirementSpec:
    """Rule-based JD parser — used when Gemini is unavailable."""
    lower = jd_text.lower()

    must_have   = [k for k in _MUST_HAVE_KEYWORDS   if k in lower]
    nice_to_have= [k for k in _NICE_TO_HAVE_KEYWORDS if k in lower]

    # Experience band
    exp_min, exp_max = 3, 10
    for m in _EXP_PATTERN.finditer(jd_text):
        g = m.groups()
        if g[0] and g[1]:
            exp_min, exp_max = int(g[0]), int(g[1])
        elif g[2]:
            exp_min = int(g[2])
            exp_max = exp_min + 5
        elif g[3]:
            exp_min = max(0, int(g[3]) - 1)
            exp_max = int(g[3]) + 3
        break

    # Seniority
    seniority = 'mid'
    for label, patterns in _SENIORITY_MAP.items():
        if any(re.search(p, lower) for p in patterns):
            seniority = label
            break

    product_required = any(w in lower for w in [
        'product company', 'product-based', 'not consulting', 'startup', 'saas', 'b2b', 'b2c'
    ])

    # Weight inference: more skill keywords → higher technical weight
    tech_weight = min(0.60, 0.35 + len(must_have) * 0.02)
    remaining   = 1.0 - tech_weight
    exp_weight  = round(remaining * 0.45, 2)
    domain_w    = round(remaining * 0.35, 2)
    behav_w     = round(1.0 - tech_weight - exp_weight - domain_w, 2)

    summary = (
        f"{seniority} engineer {exp_min}-{exp_max} years "
        + ' '.join(must_have[:8])
        + (' product company' if product_required else '')
    )

    data = {
        'must_have_skills':         must_have,
        'nice_to_have_skills':      nice_to_have,
        'experience_band':          {'min': exp_min, 'max': exp_max},
        'domain_context':           f'{seniority.title()}-level role in ML/AI infrastructure',
        'seniority_label':          seniority,
        'product_company_required': product_required,
        'inferred_weights': {
            'technical_skills': round(tech_weight, 2),
            'experience_fit':   exp_weight,
            'domain_fit':       domain_w,
            'behavioral':       behav_w,
        },
        'summary_for_embedding': summary,
    }
    logger.info('[Stage1] Used heuristic fallback. must_have=%s', must_have[:5])
    return RequirementSpec(data)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _llm_parse(jd_text: str) -> RequirementSpec | None:
    """
    Call Gemini 1.5 Flash to extract a RequirementSpec.
    Returns None on any failure so the caller can fall back.
    """
    api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        logger.info('[Stage1] No GROQ_API_KEY set — using heuristic parser.')
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"--- JOB DESCRIPTION ---\n{jd_text[:4000]}"}
            ],
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)

        # Validate weights sum to ~1.0
        w = data.get('inferred_weights', {})
        total = sum(w.values())
        if abs(total - 1.0) > 0.05:
            for k in w:
                w[k] = round(w[k] / total, 3)
            data['inferred_weights'] = w

        # Ensure summary_for_embedding is present
        if not data.get('summary_for_embedding'):
            skills_str = ' '.join(data.get('must_have_skills', [])[:6])
            band = data.get('experience_band', {})
            data['summary_for_embedding'] = (
                f"{data.get('seniority_label','senior')} engineer "
                f"{band.get('min',5)}-{band.get('max',9)} years {skills_str}"
            )

        logger.info('[Stage1] LLM parse success. must_have=%s',
                    data.get('must_have_skills', [])[:5])
        return RequirementSpec(data)

    except Exception as exc:  # noqa: BLE001
        logger.warning('[Stage1] LLM parse failed (%s) — falling back to heuristic.', exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def decompose(jd_text: str) -> RequirementSpec:
    """
    Stage 1 public interface.
    Tries LLM first, falls back to heuristic on any failure.
    """
    spec = _llm_parse(jd_text)
    if spec is None:
        spec = _heuristic_parse(jd_text)
    return spec
