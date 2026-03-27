import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(
    repo_id="adityapvdp/Predictive-Maintenance-model",
    filename="best_engine_prediction_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Engine Fault Prediction
st.title("Engine Fault Prediction App")
st.write(
    "The Engine Fault Prediction App is an internal tool that predicts whether an engine is likely to be faulty "
    "based on its operational sensor readings."
)
st.write("Enter the engine parameters below to check the predicted engine condition.")

# Collect user input
engine_rpm = st.number_input(
    "Engine RPM (engine speed in revolutions per minute)",
    min_value=0.0,
    value=791.0
)

lub_oil_pressure = st.number_input(
    "Lub Oil Pressure (lubricating oil pressure in bar/kPa)",
    min_value=0.0,
    value=3.30
)

fuel_pressure = st.number_input(
    "Fuel Pressure (fuel supply pressure in bar/kPa)",
    min_value=0.0,
    value=6.65
)

coolant_pressure = st.number_input(
    "Coolant Pressure (coolant system pressure in bar/kPa)",
    min_value=0.0,
    value=2.33
)

lub_oil_temp = st.number_input(
    "Lub Oil Temperature (lubricating oil temperature in °C)",
    min_value=0.0,
    value=77.64
)

coolant_temp = st.number_input(
    "Coolant Temperature (coolant temperature in °C)",
    min_value=0.0,
    value=78.43
)

# Create input dataframe
input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

# Set classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    result = "Faulty" if prediction == 1 else "Active / Normal"

    st.subheader("Prediction Result")
    st.write(f"**Predicted Engine Condition:** {result}")
    st.write(f"**Fault Probability:** {prediction_proba:.2%}")

    if prediction == 1:
        st.warning("The engine is likely to be in a faulty condition. Further inspection is recommended.")
    else:
        st.success("The engine is likely to be in an active/normal condition.")
