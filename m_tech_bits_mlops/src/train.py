from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# from sklearn.metrics import log_loss
# from sklearn.metrics import roc_auc_score


from sklearn.linear_model import LogisticRegression
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger  # Import loguru logger

mlflow.set_tracking_uri("http://0.0.0.0:5000")  # Replace <your-server-ip> with your actual server address
mlflow.set_experiment('exp-0')

# Load data
logger.info("Loading data from the URL.")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=columns)

# Preprocessing
logger.info("Preprocessing the data.")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

test_size = 0.25
random_state = 3242

# Split the dataset
logger.info("Splitting the dataset into training and testing sets.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

# Standardize features
logger.info("Standardizing features.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 

# Start an MLflow run
with mlflow.start_run():
    data.to_csv("./data/diabetes-dataset.csv", index=False)
    mlflow.log_artifact("./data/diabetes-dataset.csv")
    
    max_iter = 100

    mlflow.log_param("test_size", test_size)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("random_state", random_state)

    logger.info("Starting an MLflow run.")
    # Train a logistic regression model
    logger.info("Training the logistic regression model.")
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    logger.info("Making predictions on the test data.")
    predictions = model.predict(X_test_scaled)

    # Calculate metrics
    logger.info("Calculating accuracy metrics.")
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)

    precision = precision_score(y_test, predictions, average='weighted')
    mlflow.log_metric("precision", precision)

    recall = recall_score(y_test, predictions, average='weighted')
    mlflow.log_metric("recall", recall)

    f1 = f1_score(y_test, predictions, average='weighted')
    mlflow.log_metric("f1_score", f1)


    # Save the model
    logger.info("Saving the model to MLflow.")
    mlflow.sklearn.log_model(model, "model")
