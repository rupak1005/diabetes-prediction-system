"""
Diabetes Prediction System
Author: Rupak Kumar
Description: A predictive system using Support Vector Machines (SVM) to classify whether a person is diabetic based on given features.
"""

# Importing the required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Step 1: Data Collection and Analysis
try:
    diabetes_dataset = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The dataset file was not found.")
    exit()
except pd.errors.ParserError:
    print("Error: Failed to parse the dataset file.")
    exit()

# Displaying basic information about the dataset
print(f"Dataset Shape: {diabetes_dataset.shape}")
print("Sample Data:\n", diabetes_dataset.head())
print("\nStatistical Summary:\n", diabetes_dataset.describe())

# Checking for missing values
if diabetes_dataset.isnull().sum().any():
    print("Warning: The dataset contains missing values.")
    print(diabetes_dataset.isnull().sum())

# Outcome distribution
print("\nOutcome Distribution:\n", diabetes_dataset['Outcome'].value_counts())

# 0 --> Non-diabetic, 1 --> Diabetic
print("\nMean Values Grouped by Outcome:\n", diabetes_dataset.groupby('Outcome').mean())

# Step 2: Separating Features and Target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Step 3: Data Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(f"\nData Dimensions:\n Features: {X.shape}, Training Data: {X_train.shape}, Test Data: {X_test.shape}")

# Step 5: Model Training
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
print("\nModel training completed.")

# Step 6: Model Evaluation
# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Training Data Accuracy: {training_data_accuracy:.2f}")

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Test Data Accuracy: {test_data_accuracy:.2f}")

# Step 7: Making Predictions
def predict_diabetes(input_data):
    """
    Predict whether the given person is diabetic or not based on input data.
    Args:
    - input_data (tuple): Features as (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    
    Returns:
    - str: Prediction result
    """
    try:
        # Convert input data to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        
        # Standardize the input data
        std_data = scaler.transform(input_data_as_numpy_array)
        
        # Predict
        prediction = classifier.predict(std_data)
        outcome = "diabetic" if prediction[0] == 1 else "not diabetic"
        return f"The person is {outcome}."
    except Exception as e:
        return f"Error during prediction: {e}"

# Example Prediction
sample_input = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
print("\nSample Prediction:")
print(predict_diabetes(sample_input))
