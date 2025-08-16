import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")  # Make sure it's same used during training

st.title("Predictive Maintenance - NASA C-MAPSS")

user_input = st.text_input("Enter comma-separated sensor values (25 values expected):")

if st.button("Predict RUL"):
    try:
        values = [float(x.strip()) for x in user_input.split(",")]
        if len(values) != 25:
            st.error("❌ Please enter exactly 25 values.")
        else:
            # Reshape and scale
            input_array = np.array(values).reshape(1, -1)  # shape: (1, 25)
            scaled = scaler.transform(input_array)         # shape: (1, 25)
            scaled_input = scaled.reshape(1, 1, 25)         # shape: (batch, timestep, features)

            prediction = model.predict(scaled_input)
            st.success(f"✅ Predicted Remaining Useful Life (RUL): {prediction[0][0]:.2f}")
    except Exception as e:
        st.error(f"❌ Error processing input: {e}")

