from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load data from the uploaded file
file_path = 'iris.csv'
df = pd.read_csv(file_path)

# Splitting the dataset into features and target variable
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
classification_report_str = classification_report(y_test, predictions)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return f"Model Accuracy: {accuracy}<br>Classification Report:<br>{classification_report_str}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = np.array([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]])
    prediction = model.predict(user_input)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
