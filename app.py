import streamlit as st
import numpy as np
import pickle


model = pickle.load(open("logistic_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.set_page_config(page_title="Fraud Detection System", layout="wide")


st.markdown(
    """
    <h1 style="text-align:center; font-size:48px; margin-bottom:5px;">
        💳 Fraud Detection System
    </h1>
    <p style="text-align:center; color:gray; font-size:18px; margin-top:-10px;">
        Enter the Transaction Details Below
    </p>
    <hr style="border:1px solid #eee;">
    """,
    unsafe_allow_html=True
)


merchant_name = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude", format="%.6f")
long = st.number_input("Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")

trans_hour = st.slider("Transaction Hour", 0, 23, 12)
trans_day = st.slider("Transaction Day", 1, 31, 1)
trans_month = st.slider("Transaction Month", 1, 12, 1)

gender = st.selectbox("Gender", ["Female", "Male"])
card_number = st.text_input("Credit Card Number")

gender_value = 1 if gender == "Female" else 0


input_data = np.array([[
    amt,
    lat,
    long,
    merch_lat,
    merch_long,
    trans_hour,
    trans_day,
    trans_month,
    gender_value
]])


if st.button("Check For Fraud"):
    try:
        scaled = scaler.transform(input_data)
        pred = model.predict(scaled)[0]

        if pred == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Safe / Not Fraudulent")

    except Exception as e:
        st.error(f"Error: {e}")
