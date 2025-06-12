import os
import requests

# These should be set in your CI/CD env or .env file
ENDPOINT_URI = os.getenv("ENDPOINT_URI")
ENDPOINT_KEY = os.getenv("ENDPOINT_KEY")

# Example input, replace with your schema if different
input_data = {
    "input_data": {
        "columns": ["age", "income", "loan_amount"],
        "data": [[45, 80000, 5000]]
    }
}

if not ENDPOINT_URI or not ENDPOINT_KEY:
    raise ValueError("Missing ENDPOINT_URI or ENDPOINT_KEY.")

headers = {
    "Authorization": f"Bearer {ENDPOINT_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(ENDPOINT_URI, headers=headers, json=input_data)

print("Status code:", response.status_code)
print("Response:", response.json())