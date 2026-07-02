import json
import os
import logging

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert technical recruiter and resume analyzer. Given a raw resume text, extract structured information and provide a brief SWOT analysis (Strengths/Weaknesses).
Return ONLY valid JSON. Use this exact schema:
{
  "skills": ["skill1", "skill2", "skill3"],
  "experience_years": <integer total years of experience, estimate if needed>,
  "summary": "<2-3 sentence professional summary>",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["area for improvement 1", "area for improvement 2"]
}
Do not include markdown fences around the JSON."""

def analyze_resume(resume_text: str, api_key: str = None) -> dict:
    if not api_key:
        api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        logger.warning('No GROQ_API_KEY set — returning mock analysis.')
        return _mock_analysis(resume_text)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"--- RESUME ---\n{resume_text[:6000]}"}
            ],
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        return data
    except Exception as exc:
        logger.warning('LLM resume analysis failed (%s) — returning mock.', exc)
        return _mock_analysis(resume_text)

def _mock_analysis(resume_text: str) -> dict:
    return {
        "skills": ["Python", "Machine Learning", "Communication"],
        "experience_years": 3,
        "summary": "Software engineer with experience in backend development. This is a fallback mock summary since the LLM failed or no API key was provided.",
        "strengths": ["Basic technical foundation", "Clear formatting"],
        "weaknesses": ["Lacks specific metrics", "No advanced AI experience highlighted"]
    }
