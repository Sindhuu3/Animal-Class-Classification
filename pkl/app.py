import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Load paths safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "animal_classifier.pkl")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_mapping.pkl")

# -------------------------------
# Load model & class mapping (ONCE)
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    class_df = joblib.load(CLASS_MAP_PATH)
    return model, class_df

model, class_df = load_artifacts()

class_map = dict(zip(class_df["Class_Number"], class_df["Class_Type"]))

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Animal Classification", layout="centered")

st.title("🐾 Animal Classification App")
st.write("Predict the animal class using biological features")

# -------------------------------
# User Input Function
# -------------------------------
def user_input_features():
    data = {
        "hair": st.selectbox("Hair", [0, 1]),
        "feathers": st.selectbox("Feathers", [0, 1]),
        "eggs": st.selectbox("Eggs", [0, 1]),
        "milk": st.selectbox("Milk", [0, 1]),
        "airborne": st.selectbox("Airborne", [0, 1]),
        "aquatic": st.selectbox("Aquatic", [0, 1]),
        "predator": st.selectbox("Predator", [0, 1]),
        "toothed": st.selectbox("Toothed", [0, 1]),
        "backbone": st.selectbox("Backbone", [0, 1]),
        "breathes": st.selectbox("Breathes", [0, 1]),
        "venomous": st.selectbox("Venomous", [0, 1]),
        "fins": st.selectbox("Fins", [0, 1]),
        "legs": st.slider("Legs", 0, 8, 2),
        "tail": st.selectbox("Tail", [0, 1]),
        "domestic": st.selectbox("Domestic", [0, 1]),
        "catsize": st.selectbox("Cat Size", [0, 1])
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Animal Class"):
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df).max()

    st.success(f"Predicted Animal Class: **{class_map[prediction]}**")
    st.info(f"Prediction Confidence: **{confidence:.2f}**")
