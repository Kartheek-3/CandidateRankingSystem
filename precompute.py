import torch
import json
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Keywords to quickly filter out obviously irrelevant candidates
AI_KEYWORDS = {'ai', 'machine learning', 'ml', 'data', 'software', 'engineer', 'backend', 'retrieval', 'search', 'ranking', 'recommendation', 'nlp', 'llm', 'python', 'vector', 'embedding'}

def is_honeypot(cand):
    # Detect honeypots to completely drop them from our precomputed set
    # 1. Expert proficiency with 0 duration
    for skill in cand.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            return True
    return False

def is_relevant(cand):
    # 1. Experience check (3 to 18 years)
    exp = cand.get('profile', {}).get('years_of_experience', 0)
    if not (3 <= exp <= 18):
        return False
        
    # 2. Honeypot check
    if is_honeypot(cand):
        return False

    # 3. Text keyword check
    profile = cand.get('profile', {})
    text_to_search = (profile.get('headline', '') + ' ' + 
                      profile.get('current_title', '') + ' ' +
                      profile.get('summary', '')).lower()
    
    if not any(k in text_to_search for k in AI_KEYWORDS):
        return False
        
    return True

def generate_candidate_text(cand):
    # Create a dense string representing the candidate's core qualifications
    profile = cand.get('profile', {})
    skills = ", ".join([s['name'] for s in cand.get('skills', []) if s.get('proficiency') in ['advanced', 'expert']])
    
    career_texts = []
    for job in cand.get('career_history', []):
        career_texts.append(f"{job.get('title')} at {job.get('company')} ({job.get('duration_months')} months): {job.get('description', '')}")
    career_history = " | ".join(career_texts)
    
    text = (f"Title: {profile.get('current_title')}. "
            f"Experience: {profile.get('years_of_experience')} years. "
            f"Summary: {profile.get('summary')} "
            f"Key Skills: {skills}. "
            f"Career History: {career_history}")
    return text

def main():
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    input_file = 'dataset/candidates.jsonl'
    
    filtered_candidates = []
    texts_to_embed = []
    
    print("Parsing and filtering candidates...")
    start_time = time.time()
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                print(f"Processed {i} lines...")
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_relevant(cand):
                filtered_candidates.append(cand)
                texts_to_embed.append(generate_candidate_text(cand))
    
    print(f"Filtered down to {len(filtered_candidates)} candidates in {time.time() - start_time:.1f} seconds.")
    
    print("Generating embeddings...")
    start_time = time.time()
    # Embed in batches
    embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print(f"Generated embeddings in {time.time() - start_time:.1f} seconds.")
    
    print("Saving FAISS index...")
    # L2 normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # Inner product for cosine similarity
    index.add(embeddings)
    faiss.write_index(index, 'candidates.index')
    
    print("Saving metadata...")
    # Extract important structured metadata for the fast ranking script
    metadata_list = []
    for cand in filtered_candidates:
        sig = cand.get('redrob_signals', {})
        prof = cand.get('profile', {})
        
        # Check product company (heuristic: not TCS, Infosys, etc.)
        consulting_firms = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
        is_consulting_only = True
        has_product = False
        for job in cand.get('career_history', []):
            company = str(job.get('company')).lower()
            if company not in consulting_firms:
                has_product = True
                is_consulting_only = False
        
        # Compile all skills into a lowercase string for fast exact-match scanning
        all_skills = " ".join([s['name'].lower() for s in cand.get('skills', [])])
        
        metadata_list.append({
            'candidate_id': cand['candidate_id'],
            'years_of_experience': prof.get('years_of_experience', 0),
            'current_title': prof.get('current_title', ''),
            'is_consulting_only': is_consulting_only,
            'recruiter_response_rate': sig.get('recruiter_response_rate', 0),
            'last_active_date': sig.get('last_active_date', ''),
            'all_skills': all_skills,
            'profile_completeness_score': sig.get('profile_completeness_score', 0),
            'interview_completion_rate': sig.get('interview_completion_rate', 0)
        })
        
    df = pd.DataFrame(metadata_list)
    df.to_parquet('candidates_metadata.parquet')
    print("Pre-computation complete! Saved candidates.index and candidates_metadata.parquet.")

if __name__ == '__main__':
    main()
