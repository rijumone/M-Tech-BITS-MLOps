import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # Import joblib for saving the model

# Load the dataset (assuming it's already loaded as a pandas DataFrame)
def train_churn_model(df):
    # Drop non-essential columns
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # Define hyperparameters grid
    param_grid = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 150],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Print best parameters from GridSearchCV
    print("Best Hyperparameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model using joblib
    joblib.dump(best_model, 'best_xgboost_model.joblib')

    # Make predictions with the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Return the trained model and label encoders
    return best_model, label_encoders


if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("./data/Churn_Modelling.csv")
    model, encoders = train_churn_model(df)

