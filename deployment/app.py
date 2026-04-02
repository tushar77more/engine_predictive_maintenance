
import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
#  Page Config
# -------------------------------
st.set_page_config(page_title="Engine Maintenance Predictor", layout="wide")
st.title("Engine Predictive Maintenance")

# -------------------------------
#  Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_filename = "GradientBoosting_best_model.pkl"

    if os.path.exists(model_filename):
        try:
            return joblib.load(model_filename)
        except Exception as e:
            st.error(f" Error loading model: {e}")
            return None
    else:
        st.error(f" Model file '{model_filename}' not found!")
        st.info(f"Available files: {os.listdir('.')}")
        return None

model = load_model()

# -------------------------------
#  Input UI
# -------------------------------
st.header("Enter Engine Sensor Details")

col1, col2, col3 = st.columns(3)

with col1:
    Engine_RPM = st.number_input("Engine RPM", 500, 5000, 1500)

with col2:
    Lub_Oil_Pressure = st.number_input("Lub Oil Pressure", 0.0, 10.0, 3.5)

with col3:
    Fuel_Pressure = st.number_input("Fuel Pressure", 0.0, 10.0, 5.2)

col4, col5, col6 = st.columns(3)

with col4:
    Coolant_Pressure = st.number_input("Coolant Pressure", 0.0, 10.0, 2.1)

with col5:
    Lub_Oil_Temp = st.number_input("Lub Oil Temp (°C)", 0, 150, 80)

with col6:
    Coolant_Temp = st.number_input("Coolant Temp (°C)", 0, 150, 95)

# -------------------------------
#  Prediction
# -------------------------------
if st.button("Predict Engine Condition"):

    if model:
        data = {
            "Engine_RPM": Engine_RPM,
            "Lub_Oil_Pressure": Lub_Oil_Pressure,
            "Fuel_Pressure": Fuel_Pressure,
            "Coolant_Pressure": Coolant_Pressure,
            "Lub_Oil_Temp": Lub_Oil_Temp,
            "Coolant_Temp": Coolant_Temp
        }

        input_df = pd.DataFrame([data])

        try:
            # Ensure correct feature order (VERY IMPORTANT)
            if hasattr(model, "feature_names_in_"):
                input_df = input_df[list(model.feature_names_in_)]

            prediction = model.predict(input_df)[0]

            st.divider()

            if prediction == 1:
                st.error("Engine is likely FAULTY!")
                st.warning("Immediate maintenance recommended.")
            else:
                st.success("Engine is operating normally.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    else:
        st.error("Model not loaded properly.")
