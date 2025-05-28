import requests

def test_health_check():
    response = requests.get("http://localhost:8000")
    assert response.status_code == 200
    assert "message" in response.json()


