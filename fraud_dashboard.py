import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained model
model = joblib.load('xgb_model.pkl')

# Set page config
st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="wide")

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("<style>body { background-color: #0e1117; color: #fafafa; }</style>", unsafe_allow_html=True)

# Initialize session state
if 'predicted_transactions' not in st.session_state:
    st.session_state.predicted_transactions = []

# Sidebar navigation
st.sidebar.markdown("📑 **Navigation**")
section = st.sidebar.radio("Go to", ["🏠 Overview", "🔍 Predict", "📬 Upload & Monitor", "📊 Reports"])

# Main App Interface
st.title("🔍 Real-Time Transaction Prediction")

if section == "🏠 Overview":
    st.subheader("📄 Project Overview")
    st.write("""
        This AI-powered dashboard uses a machine learning model trained on transaction data to detect fraudulent activities. 
        You can test real-time predictions, monitor transactions, and view fraud detection reports.
    """)

elif section == "🔍 Predict":
    st.subheader("⚡ Real-Time Transaction Prediction")
    
    amount = st.number_input("Transaction Amount", value=5000.0)
    tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
    old_balance = st.number_input("Old Balance", value=10000.0)
    new_balance = st.number_input("New Balance", value=500.0)

    if st.button("🧠 Predict"):
        try:
            type_map = {"TRANSFER": 0, "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "DEBIT": 4}
            input_data = pd.DataFrame([[
                type_map[tx_type], amount, old_balance, new_balance,
                old_balance - amount, new_balance + amount,
                int(old_balance == 0), int(new_balance == 0)
            ]], columns=[
                "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                "diffOrig", "estNewDest", "flagOldZero", "flagNewZero"
            ])
            prediction = model.predict(input_data)[0]
            result = "FRAUDULENT ❌" if prediction == 1 else "LEGIT ✅"
            color = "red" if prediction == 1 else "green"
            st.success(f"Prediction: {result}")

            st.session_state.predicted_transactions.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Amount": amount,
                "Type": tx_type,
                "Old Balance": old_balance,
                "New Balance": new_balance,
                "Prediction": result
            })
        except Exception as e:
            st.error(f"Something went wrong: {e}")

elif section == "📬 Upload & Monitor":
    st.subheader("📬 Monitored Transactions")
    st.write("Log of all tested transactions.")

    if st.session_state.predicted_transactions:
        df_logs = pd.DataFrame(st.session_state.predicted_transactions)
        st.dataframe(df_logs)
    else:
        st.info("No transactions monitored yet.")

elif section == "📊 Reports":
    st.subheader("📊 Model Performance Report")
    st.write("This section will soon include visualizations of performance metrics and insights from the full test set.")
    st.info("Coming soon!")






