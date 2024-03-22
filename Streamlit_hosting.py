import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

red_data = pd.read_csv('wine_quality\winequality-red.csv', delimiter = ';')

X_red = red_data.drop('quality', axis = 1)
y_red = red_data['quality']


# Split the dataset into training and validation sets
X_red_train, X_red_valid, y_red_train, y_red_valid = train_test_split(X_red, y_red, test_size=0.2, random_state=21)

# Define sliders for hyperparameters
max_depth = st.sidebar.slider('Max Depth', 30, 90, step=10, value=30)
n_estimators = st.sidebar.slider('Number of Estimators', 200, 500, step=100, value=200)

# Train the Random Forest classifier
rf_classifier_red = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=21)
rf_classifier_red.fit(X_red_train, y_red_train)

# Evaluate the model on the validation set
y_pred = rf_classifier_red.predict(X_red_valid)
accuracy = accuracy_score(y_red_valid, y_pred)

# Display validation accuracy
st.write(f"Validation Accuracy (n_estimators={n_estimators}, max_depth={max_depth}): {accuracy}")

# Display the best model
st.write(f"\nBest Model for red-wine data (n_estimators={n_estimators}, max_depth={max_depth}) Validation Accuracy: {accuracy}")
