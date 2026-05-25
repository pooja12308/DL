import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.title("🚗 Car Price Prediction AI")

st.write("Enter the car details to predict the price.")

# Load model
model = load_model("car_price_model.keras")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature columns
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# User Inputs
symboling = st.number_input("Symboling", -3, 3, 0)
wheelbase = st.number_input("Wheelbase")
carlength = st.number_input("Car Length")
carwidth = st.number_input("Car Width")
carheight = st.number_input("Car Height")
curbweight = st.number_input("Curb Weight")
enginesize = st.number_input("Engine Size")
boreratio = st.number_input("Bore Ratio")
stroke = st.number_input("Stroke")
compressionratio = st.number_input("Compression Ratio")
horsepower = st.number_input("Horsepower")
peakrpm = st.number_input("Peak RPM")
citympg = st.number_input("City MPG")
highwaympg = st.number_input("Highway MPG")

# Create dataframe
input_data = pd.DataFrame([[symboling,wheelbase,carlength,carwidth,carheight,
                            curbweight,enginesize,boreratio,stroke,
                            compressionratio,horsepower,peakrpm,
                            citympg,highwaympg]],
                           columns=[
                           'symboling','wheelbase','carlength','carwidth',
                           'carheight','curbweight','enginesize','boreratio',
                           'stroke','compressionratio','horsepower','peakrpm',
                           'citympg','highwaympg'])

# Add missing columns
for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[features]

# Scale
input_scaled = scaler.transform(input_data)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Car Price: ${prediction[0][0]:,.2f}")