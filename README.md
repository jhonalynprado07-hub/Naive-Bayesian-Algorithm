# Naive-Bayesian-Algorithm
Asynchronous Activity

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'StudyHours': [2, 3, 5, 7, 8, 1, 4, 6, 2, 9],
    'Attendance': [60, 70, 80, 90, 95, 50, 75, 85, 65, 98],
    'SleepHours': [5, 6, 7, 6, 8, 4, 6, 7, 5, 8],
    'Result': ['Fail','Fail','Pass','Pass','Pass','Fail','Pass','Pass','Fail','Pass']
}

df = pd.DataFrame(data)

# Feature Names
X = df[['StudyHours', 'Attendance', 'SleepHours']]
y = df['Result']


# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Simulation
test_student = pd.DataFrame(
    [[6, 85, 7]],
    columns=['StudyHours', 'Attendance', 'SleepHours']
)
prediction = model.predict(test_student)
probability = model.predict_proba(test_student)

print("\n--- Simulation Result ---")
print("Student Data: Study=6hrs, Attendance=85%, Sleep=7hrs")
print("Probability [Fail]:", round(probability[0][0], 3))
print("Probability [Pass]:", round(probability[0][1], 3))
print("Final Prediction:", prediction[0])
