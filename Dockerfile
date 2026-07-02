FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Assume dataset/candidates.jsonl is mounted or passed during execution.
# To run ranking: python rank.py --candidates dataset/candidates.jsonl --out submission.csv

CMD ["python", "rank.py", "--candidates", "dataset/candidates.jsonl", "--out", "submission.csv"]
