import requests

# URL of the Flask app (change if it's running on a different host/port)
url = "http://localhost:5002/predict"

# Test data for prediction
test_data = [
    {
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
    }
]

# Sending the POST request to the /predict endpoint
response = requests.post(url, json=test_data)

# Check if the request was successful
if response.status_code == 200:
    print("Prediction successful!")
    print("Response:", response.json())
else:
    print("Error:", response.status_code)
    print(response.json())
