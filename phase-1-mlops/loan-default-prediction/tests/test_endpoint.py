import os
import requests

endpoint = os.getenv("ENDPOINT_URI")
key = os.getenv("ENDPOINT_KEY")

resp = requests.post(endpoint, headers={"Authorization": f"Bearer {key}"}, json={"data": [1,2,3]})
assert resp.status_code == 200
print(resp.json())
