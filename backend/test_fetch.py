import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_ROLE_KEY'))

response = supabase.table('candidates_vector').select('metadata').limit(5).execute()
for row in response.data:
    print(row)
