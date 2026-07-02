import requests

url = 'http://127.0.0.1:5001/api/rank'
payload = {
    'jd': 'I need data scientists',
    'batch_id': 'test-batch-id'
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
