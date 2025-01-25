import pytest
from collections import OrderedDict
from m_tech_bits_mlops.src.app import app


@pytest.fixture
def client():
    # Create a test client for the Flask app
    with app.test_client() as client:
        yield client


# Test the health check endpoint
def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.data == b"Healthy"


# Test the predict endpoint with valid input
def test_predict(client):
    # Sample test data
    test_data = [
        OrderedDict({
            "CreditScore": 200,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42.0,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1.0,
            "IsActiveMember": 0.0,
            "EstimatedSalary": 1018.88
        })
    ]

    # Make the POST request to /predict
    response = client.post('/predict', json=test_data)
    # Assert status code and check for valid prediction response
    assert response.status_code == 200
