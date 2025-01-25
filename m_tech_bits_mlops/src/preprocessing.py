import pandas as pd

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(url, names=columns)


# Feature Engineering: Adding AgeBMIIndex
df['AgeBMIIndex'] = df['BMI'] / (df['Age'] + 1)

# Display the first few rows to verify the new feature
print(df.head())

# Evaluate correlation with Outcome
correlation = df.corr()['Outcome']['AgeBMIIndex']
print(f"Correlation between AgeBMIIndex and Outcome: {correlation}")

# Save the updated dataset
df.to_csv("./data/diabetes-dataset.csv", index=False)
