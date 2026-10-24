import os
import sys
import time
import requests
from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "LEARNINGDNA_LLM_URL",
    ""
).rstrip("/")

API_KEY = os.getenv(
    "LEARNINGDNA_API_KEY",
    ""
)

if not API_URL or not API_KEY:
    print("ERROR: API configuration is missing.")
    print("Set LEARNINGDNA_LLM_URL and LEARNINGDNA_API_KEY.")
    sys.exit(1)

HEADERS = {
    "X-API-Key": API_KEY
}

# --------------------------------------------------
# Test 1: Server health
# --------------------------------------------------

print("\n[TEST 1] Checking server health...")

try:
    start = time.perf_counter()

    response = requests.get(
        f"{API_URL}/health",
        timeout=15
    )

    elapsed = time.perf_counter() - start

    print("HTTP Status:", response.status_code)
    print("Response:", response.text)
    print(f"Latency: {elapsed:.2f} seconds")

    response.raise_for_status()

    print("PASS: Server is reachable.")

except requests.RequestException as exc:
    print("FAIL: Could not reach the server.")
    print("Details:", exc)
    sys.exit(1)

# --------------------------------------------------
# Test 2: Authentication
# --------------------------------------------------

print("\n[TEST 2] Checking API authentication...")

try:
    response = requests.post(
        f"{API_URL}/generate",
        headers={"X-API-Key": "invalid-test-key"},
        json={
            "prompt": "Hello",
            "context": "Test",
            "max_new_tokens": 20,
            "temperature": 0.0
        },
        timeout=15
    )

    if response.status_code == 401:
        print("PASS: Invalid API key correctly rejected.")
    else:
        print("WARNING: Unexpected status:", response.status_code)
        print(response.text)

except requests.RequestException as exc:
    print("FAIL:", exc)

# --------------------------------------------------
# Test 3: LLM generation
# --------------------------------------------------

print("\n[TEST 3] Testing Phi-3 generation...")

payload = {
    "prompt": "Explain inheritance in object-oriented programming.",
    "context": (
        "Inheritance allows a derived class to acquire "
        "properties and methods from a base class."
    ),
    "max_new_tokens": 150,
    "temperature": 0.2
}

try:
    start = time.perf_counter()

    response = requests.post(
        f"{API_URL}/generate",
        headers=HEADERS,
        json=payload,
        timeout=(10, 180)
    )

    elapsed = time.perf_counter() - start

    print("HTTP Status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    print("\nGenerated Answer:")
    print(result.get("response", "No response returned."))

    print("\nServer generation time:",
          result.get("generation_seconds", "Unavailable"),
          "seconds")

    print(f"Total request time: {elapsed:.2f} seconds")

    print("\nPASS: LLM generation succeeded.")

except requests.RequestException as exc:
    print("FAIL: Generation request failed.")
    print("Details:", exc)

    if getattr(exc, "response", None) is not None:
        print("Response body:", exc.response.text)

    sys.exit(1)

print("\nAll critical API tests completed.")