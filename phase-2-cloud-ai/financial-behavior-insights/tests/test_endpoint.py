import requests
import json
import os
import sys
from dotenv import load_dotenv

def load_env_vars():
    load_dotenv()
    endpoint_uri = os.getenv("ENDPOINT_URI")
    endpoint_key = os.getenv("ENDPOINT_KEY")
    if not endpoint_uri or not endpoint_key:
        print("Error: ENDPOINT_URI and ENDPOINT_KEY must be set in the environment or .env file.")
        sys.exit(1)
    return endpoint_uri, endpoint_key

def build_headers(endpoint_key):
    return {
        "Authorization": f"Bearer {endpoint_key}",
        "Content-Type": "application/json"
    }

def build_payload():
    # TODO: Adjust payload as per your model's requirements
    return {
        "input_data": [
            [
                -0.29515013823277825,1.654052565802118,-1.5888898783976668,-0.9565347175325561,0.0,1.4849748866948589,0.0,0.0,1.0,1.0,0.0,0.0
            ]
        ]
    }

def test_connection(endpoint_uri, headers):
    try:
        response = requests.get(endpoint_uri, headers=headers, timeout=5)
        print("Connection test status code:", response.status_code)
        if response.status_code == 200:
            print("Connection successful.")
        else:
            print("Connection failed or endpoint does not support GET requests.")
    except requests.exceptions.RequestException as e:
        print(f"Connection test failed: {e}")
        sys.exit(1)

def main():
    endpoint_uri, endpoint_key = load_env_vars()
    headers = build_headers(endpoint_key)

    # Test connection first
    test_connection(endpoint_uri, headers)

    payload = build_payload()
    try:
        response = requests.post(endpoint_uri, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        try:
            print("Response:", json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            print("Response is not valid JSON:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()