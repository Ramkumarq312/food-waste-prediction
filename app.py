"""
app.py - Streamlit UI for the Food Waste Prediction System.
Run:  streamlit run app.py   (after running python train_model.py)
"""

import os
import pickle

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EVENTS = ["Regular", "Festival", "Exam", "Holiday", "Special Event"]
WEATHER = ["Sunny", "Rainy", "Cloudy"]
HIGH_WASTE_LIMIT_KG = 20

st.set_page_config(page_title="Food Waste Prediction System", page_icon="🍽️", layout="centered")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ---------------- Header ----------------
st.title("🍽️ Food Waste Prediction System")
st.caption("Predict food waste in a college hostel / mess / canteen using a Random Forest model.")
st.divider()

if not os.path.exists(MODEL_PATH):
    st.error("model.pkl not found. Please run `python train_model.py` first.")
    st.stop()

model = load_model()

# ---------------- Inputs ----------------
st.subheader("Enter Details")

col1, col2 = st.columns(2)
with col1:
    food_prepared = st.number_input(
        "Food Prepared (Kg)", min_value=10.0, max_value=1000.0, value=200.0, step=5.0
    )
    expected_attendance = st.number_input(
        "Expected Attendance", min_value=1, max_value=5000, value=500, step=10
    )
    actual_attendance = st.number_input(
        "Actual Attendance", min_value=0, max_value=5000, value=400, step=10
    )
with col2:
    day_of_week = st.selectbox("Day of Week", DAYS)
    event_type = st.selectbox("Event Type", EVENTS)
    weather = st.selectbox("Weather", WEATHER)

st.write("")

# ---------------- Prediction ----------------
if st.button("Predict Food Waste", type="primary", use_container_width=True):
    attendance_ratio = actual_attendance / max(expected_attendance, 1)

    input_df = pd.DataFrame(
        [
            {
                "food_prepared": food_prepared,
                "expected_attendance": expected_attendance,
                "actual_attendance": actual_attendance,
                "attendance_ratio": attendance_ratio,
                "day_of_week": day_of_week,
                "event_type": event_type,
                "weather": weather,
            }
        ]
    )

    raw_prediction = float(model.predict(input_df)[0])

    # Clamp: never negative and never more than the food prepared
    waste_kg = min(max(raw_prediction, 0.0), float(food_prepared))
    waste_percent = (waste_kg / food_prepared) * 100

    st.divider()
    st.subheader("Prediction Result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Waste", f"{waste_kg:.2f} Kg")
    m2.metric("Waste Percentage", f"{waste_percent:.2f} %")
    m3.metric("Attendance Rate", f"{attendance_ratio * 100:.1f} %")

    st.success(f"Predicted Food Waste: {waste_kg:.2f} Kg ({waste_percent:.2f}% of food prepared)")

    if waste_kg > HIGH_WASTE_LIMIT_KG:
        st.warning("High Waste! Reduce food quantity")
    else:
        st.info("Low Waste. Good Planning!")

    if attendance_ratio < 0.8:
        st.info(
            "Note: Attendance is below 80% of the expected number, "
            "which is a major cause of sudden waste spikes."
        )

st.divider()
st.caption("Tech Project Management and AI Implementation | Food Waste Prediction System")
