import os
import json
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from agents.stage2_prefilter import build_candidate_text, _get_model

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("Error: Supabase credentials not found in .env")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
model = _get_model()

print("Loading local candidates...")
dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'candidates.jsonl')

candidates = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            cand = json.loads(line)
            candidates.append(cand)

print(f"Found {len(candidates)} candidates. Uploading to Supabase...")

inserted = 0
for cand in candidates:
    # Ensure candidate has an ID
    if 'id' in cand and 'candidate_id' not in cand:
        cand['candidate_id'] = cand.pop('id')
    candidate_id = cand.get('candidate_id', str(uuid.uuid4()))
    cand['candidate_id'] = candidate_id
    
    text = build_candidate_text(cand)
    vec = model.encode([text], convert_to_numpy=True)[0].tolist()
    
    supabase.table('candidates_vector').upsert({
        'candidate_id': candidate_id,
        'metadata': cand,
        'embedding': vec
    }, on_conflict='candidate_id').execute()
    
    inserted += 1
    if inserted % 10 == 0:
        print(f"Uploaded {inserted}/{len(candidates)}...")

print("Successfully seeded Supabase with initial dataset!")
