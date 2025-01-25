import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # Import joblib for saving the model
import mlflow
from loguru import logger


mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment('exp-1')



# Load the dataset (assuming it's already loaded as a pandas DataFrame)
def train_churn_model(df):
    test_size = 0.25
    random_state = 3242
    
    # Start an MLflow run
    with mlflow.start_run():
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
    
        # Drop non-essential columns
        mlflow.log_artifact("./data/Churn_Modelling.csv")
        df = df.drop(["RowNumber","CustomerId", "Surname"], axis=1)

        # Encode categorical variables
        label_encoders = {}
        for column in ["Geography", "Gender"]:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        # Define features and target variable
        X = df.drop("Exited", axis=1)
        y = df["Exited"]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size, random_state=random_state,
        )

        # Initialize the XGBoost classifier
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        max_depth = [3, 6, 10]
        mlflow.log_param("max_depth", max_depth)
        learning_rate = [0.01, 0.1, 0.2, 0.5]
        mlflow.log_param("learning_rate", learning_rate)
        n_estimators = [50, 100, 150]
        mlflow.log_param("n_estimators", n_estimators)
        subsample = [0.8, 0.9, 1.0]
        mlflow.log_param("subsample", subsample)
        colsample_bytree = [0.8, 0.9, 1.0]
        mlflow.log_param("colsample_bytree", colsample_bytree)

        # Define hyperparameters grid
        param_grid = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
        }

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Print best parameters from GridSearchCV
        logger.info(f"Best Hyperparameters: {grid_search.best_params_}")

        # Get the best model
        best_model = grid_search.best_estimator_

        # Save the best model using joblib
        joblib.dump(best_model, './best_xgboost_model.joblib')

        # Save label encoders
        joblib.dump(label_encoders, './label_encoders.joblib')


        # Make predictions with the best model
        y_pred = best_model.predict(X_test)

        
        # Evaluate the model
        _accuracy_score = accuracy_score(y_test, y_pred)
        mlflow.log_metric("_accuracy_score", _accuracy_score)
        logger.info(f"Accuracy: {_accuracy_score}")
        
        _classification_report = classification_report(y_test, y_pred)
        # mlflow.log_metric("_classification_report", _classification_report)
        logger.info(f"Classification Report:\n {_classification_report}")

    # Return the trained model and label encoders
    return best_model, label_encoders


if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("./data/Churn_Modelling.csv")
    model, encoders = train_churn_model(df)

