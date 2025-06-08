import requests
import json
import os

# Replace with your actual scoring URI and primary key
# scoring_uri = os.getenv("SCORING_URI")
# primary_key = os.getenv("PRIMARY_KEY")

scoring_uri = "https://loandef-ep-cecc9b.australiaeast.inference.ml.azure.com/score"
primary_key = "FVHhcXPecc9W1eYt9gqtjyzGlk7DAqdu0BlrhvsDecheIRwOG1QEJQQJ99BFAAAAAAAAAAAAINFRAZML3YGj"
# Adjust the input structure to match your model!
data = {
    "input_data": {
        "columns": ["age", "income", "loan_amount"],
        "data": [
            [35, 85000, 12000],    # Example row 1
            [50, 50000, 10000]     # Example row 2
        ]
    }
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {primary_key}"
}

response = requests.post(scoring_uri, headers=headers, data=json.dumps(data))
print("Status code:", response.status_code)
print("Prediction:", response.json())