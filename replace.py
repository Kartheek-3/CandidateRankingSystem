import os, re
path = r'frontend/src/components/ChatInterface.jsx'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'const BACKEND_URL =' not in content:
    content = re.sub(r'import axios from \'axios\';', 
                     'import axios from \'axios\';\n\nconst BACKEND_URL = import.meta.env.VITE_BACKEND_URL || \'http://127.0.0.1:5001\';', 
                     content)

content = content.replace("'http://127.0.0.1:5001/api/", "" + "/api/")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
