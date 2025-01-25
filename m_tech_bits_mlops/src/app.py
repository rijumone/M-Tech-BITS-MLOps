from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('./best_xgboost_model.joblib')
label_encoders = joblib.load('./label_encoders.joblib')

# Health check endpoint


@app.route('/health', methods=['GET'])
def health_check():
    return "Healthy", 200

# Prediction endpoint


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(data)

        # Drop non-essential columns
        df = df.drop(columns=['RowNumber', 'CustomerId',
                     'Surname', 'Exited'], errors='ignore')

        # Encode categorical columns using label encoders
        for column in ['Geography', 'Gender']:
            if column in df.columns:
                df[column] = label_encoders[column].transform(df[column])

        # Prepare the features for prediction (make sure all columns are numeric)
        X = df

        # Make predictions using the loaded model
        predictions = model.predict(X)
        # Get probability of class 1 (churn)
        probabilities = model.predict_proba(X)[:, 1]

        # Prepare the response
        response = []
        for i in range(len(predictions)):
            response.append({
                # True if churn (1), False if not churn (0)
                "chrun_prediction": bool(predictions[i]),
                # Probability of churn (class 1)
                "chrun_probability": float(probabilities[i])
            })

        # Return predictions and confidence as a JSON response
        return jsonify({"predictions": response}), 200

    except Exception as e:
        # Handle any errors
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
