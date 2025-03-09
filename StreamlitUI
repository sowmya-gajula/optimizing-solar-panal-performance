import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model

# Load model and scaler
model = load_model("solar_angle_model.h5")
scaler = joblib.load("scaler.pkl")

# UI Title
st.title("Solar Angle Prediction")

# User Input Form
st.sidebar.header("Input Parameters")
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=30.0)
humidity = st.sidebar.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
clearsky_dhi = st.sidebar.number_input("Clearsky DHI (W/m²)", min_value=0.0, value=600.0)
clearsky_dni = st.sidebar.number_input("Clearsky DNI (W/m²)", min_value=0.0, value=1000.0)
clearsky_ghi = st.sidebar.number_input("Clearsky GHI (W/m²)", min_value=0.0, value=1600.0)
dew_point = st.sidebar.number_input("Dew Point (°C)", min_value=-50.0, max_value=50.0, value=18.0)
pressure = st.sidebar.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1015.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, value=5.0)
topo_azimuth = st.sidebar.number_input("Topocentric Azimuth Angle (°)", min_value=0.0, max_value=360.0, value=180.0)
topo_zenith = st.sidebar.number_input("Topocentric Zenith Angle (°)", min_value=0.0, max_value=90.0, value=45.0)

# Prepare input
input_data = pd.DataFrame([[topo_zenith, topo_azimuth, temperature, clearsky_dhi, clearsky_dni,
                            clearsky_ghi, humidity, wind_speed, pressure, dew_point]],
                          columns=['Topocentric zenith angle', 'Top. azimuth angle (eastward from N)',
                                   'Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI',
                                   'Relative Humidity', 'Wind Speed', 'Pressure', 'Dew Point'])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Solar Angles"):
    prediction = model.predict(scaled_input)[0]
    
    # Clip values
    azimuth = np.clip(prediction[0], 0, 360)
    zenith = np.clip(prediction[1], 0, 90)
    tilt = np.clip(prediction[2], 0, 90)

    # Convert azimuth to direction
    def azimuth_to_direction(azimuth):
        directions = ['North', 'North-East', 'East', 'South-East', 'South', 'South-West', 'West', 'North-West']
        return directions[int((azimuth + 22.5) // 45) % 8]

    azimuth_direction = azimuth_to_direction(azimuth)

    # Display results
    st.success(f"Predicted Azimuth: {azimuth:.2f}° ({azimuth_direction})")
    st.success(f"Predicted Zenith: {zenith:.2f}°")
    st.success(f"Predicted Tilt: {tilt:.2f}°")
