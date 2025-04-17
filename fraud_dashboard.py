
import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="centered")
st.title("🔍 Real-Time Transaction Prediction")

# Load the trained XGBoost model
xgb_model = joblib.load("xgb_model.pkl")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
old_balance = st.number_input("Old Balance", min_value=0.0, value=0.0)
new_balance = st.number_input("New Balance", min_value=0.0, value=0.0)

# Predict button
if st.button("Predict"):
    # Encode transaction type
    type_code = 0
    if transaction_type == "TRANSFER":
        type_code = 1
    elif transaction_type == "CASH_OUT":
        type_code = 2

    # Prepare input for model
    input_df = pd.DataFrame([{
        'type': type_code,
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance
    }])

    # Run prediction
    prediction = xgb_model.predict(input_df)[0]

    if prediction == 1:
        st.error("❌ This transaction is FRAUDULENT.")
    else:
        st.success("✅ This transaction is LEGITIMATE.")
