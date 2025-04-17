import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained XGBoost model
with open("xgb_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)

# Streamlit page config
st.set_page_config(page_title="AI Fraud Detection", layout="centered")
st.title("🛡️ AI-Powered Fraud Detection Dashboard")
st.markdown("---")

# --- Section 1: Real-Time Transaction Prediction ---
st.subheader("🔍 Real-Time Transaction Prediction")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0, format="%.2f")
type_input = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
oldbalance = st.number_input("Old Balance", min_value=0.0, step=1.0, format="%.2f")
newbalance = st.number_input("New Balance", min_value=0.0, step=1.0, format="%.2f")

# Prepare data
type_map = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3
}
type_code = type_map[type_input]

if st.button("🔎 Predict"):
    input_df = pd.DataFrame([[type_code, amount, oldbalance, newbalance]],
                            columns=["type", "amount", "oldbalanceOrg", "newbalanceOrig"])
    prediction = xgb_model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 This transaction is FRAUDULENT.")
    else:
        st.success("✅ This transaction is LEGITIMATE.")

# --- Section 2: Downloadable Report Placeholder ---
st.markdown("---")
st.subheader("📄 Fraud Detection Report")

st.info("The system can generate a downloadable fraud detection report from historical or uploaded transactions. (Feature under construction)")

# Example: You can uncomment and use the below when ready
# csv = df.to_csv(index=False).encode("utf-8")
# st.download_button("Download Report", csv, "fraud_report.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.caption("© 2025 AI Fraud Detection System · Powered by XGBoost · Designed for Secure Payments")

