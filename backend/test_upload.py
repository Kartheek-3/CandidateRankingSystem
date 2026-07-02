import requests

url = 'http://127.0.0.1:5000/api/upload_candidates'
files = {'file': ('test_candidates.csv', open('../test_candidates.csv', 'rb'), 'text/csv')}

try:
    response = requests.post(url, files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
