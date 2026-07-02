import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv('SUPABASE_DB_URL')
# Wait, I don't have SUPABASE_DB_URL, only SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
# Let me query via REST API using the supabase client.
# The Supabase Python client can execute an RPC if there is an exec_sql RPC, but there isn't.
# I can try to connect to the DB via connection string if they have it in .env.
with open('.env', 'r') as f:
    print(f.read())
