import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
import json

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_ROLE_KEY'))
model = SentenceTransformer('all-MiniLM-L6-v2')

query_text = "mid engineer 3-10 years "
query_vec = model.encode([query_text], convert_to_numpy=True)[0].tolist()

# Get the batch_ids available in the DB
res = supabase.table('candidates_vector').select('metadata->>batch_id').execute()
batch_ids = list(set([r['batch_id'] for r in res.data if r['batch_id']]))
print(f"Available batch_ids: {batch_ids}")

if batch_ids:
    test_batch_id = batch_ids[0]
    rpc_payload = {
        'query_embedding': query_vec,
        'match_threshold': 0.0,
        'match_count': 100,
        'filter_batch_id': test_batch_id
    }
    response = supabase.rpc('match_candidates', rpc_payload).execute()
    print(f"Candidates found for batch {test_batch_id}: {len(response.data)}")
else:
    print("No batch_ids found!")
