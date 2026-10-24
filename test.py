import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.environ["LEARNINGDNA_LLM_URL"].rstrip("/") + "/generate"
api_key = os.environ["LEARNINGDNA_API_KEY"]

payload = {
    "prompt": (
        'Return exactly this JSON object and nothing else: '
        '{"concepts":[],"relationships":[],"learning_objectives":[]}. '
        'Do not use Markdown or code fences.'
    ),
    "context": "",
    "max_new_tokens": 100,
    "temperature": 0.0
}

response = requests.post(
    url,
    headers={"X-API-Key": api_key},
    json=payload,
    timeout=120
)

print("HTTP status:", response.status_code)
response.raise_for_status()

data = response.json()
print("\nRaw response field:")
print(repr(data.get("response")))