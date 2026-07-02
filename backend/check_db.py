import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(url, key)

response = supabase.table('candidates_vector').select('candidate_id', count='exact').limit(1).execute()
print(f"Total rows in Supabase: {response.count}")
