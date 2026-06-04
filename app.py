import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("Student Performance.csv")

# Encode Target Variable
encoder = LabelEncoder()
data["Result"] = encoder.fit_transform(data["Result"])

# Features and Target
X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Result"]

# Train Model
model = LogisticRegression()
model.fit(X, y)

# App Title
st.title("🎓 Student Performance Prediction System")

st.write(
    "Predict whether a student will Pass or Fail based on Study Hours, Attendance, and Previous Scores."
)

# User Inputs
study_hours = st.slider("Study Hours", 1, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.slider("Previous Score", 0, 100, 60)

# Prediction Button
if st.button("Predict Result"):
    prediction = model.predict(
        [[study_hours, attendance, previous_score]]
    )[0]

    if prediction == 1:
        st.success("Prediction: PASS")
    else:
        st.error("Prediction: FAIL")

# Display Dataset
if st.checkbox("Show Dataset"):
    st.dataframe(data)