import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
# Initialize session state to store transaction log
if "predicted_transactions" not in st.session_state:
    st.session_state.predicted_transactions = []


# Load the trained model
model = joblib.load("xgb_model.pkl")

# Set page configuration
st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# Sidebar navigation
st.sidebar.title("🗂️ Navigation")
selected = st.sidebar.radio("Go to", ["🏠 Overview", "🔍 Predict", "📤 Upload & Monitor", "📊 Reports"])

# DARK MODE TOGGLE
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# Apply dark mode background
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #1e1e1e; color: white; }
        .stApp { background-color: #1e1e1e; }
        </style>
    """, unsafe_allow_html=True)

# =============================
# 🏠 Overview Page
# =============================
if selected == "🏠 Overview":
    st.title("💡 Project Overview")
    st.markdown("""
        Welcome to the **AI-Powered Fraud Detection System**.  
        This dashboard provides a smart and interactive way to:
        - Monitor transactions in real-time  
        - Detect fraudulent behavior using AI models  
        - Explore detailed reports and metrics  
    """)
    st.image("https://img.freepik.com/free-vector/artificial-intelligence-illustration_52683-101910.jpg", width=600)

# =============================
# 🔍 Predict Page
# =============================
st.subheader("🔍 Real-Time Transaction Prediction")

# User Inputs
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
old_balance = st.number_input("Old Balance", min_value=0.0, format="%.2f")
new_balance = st.number_input("New Balance", min_value=0.0, format="%.2f")

if st.button("🔮 Predict"):
    try:
        # Convert transaction type to numeric
        type_dict = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
        type_num = type_dict.get(transaction_type.upper(), 0)

        # Create input with all 8 expected features
        input_data = pd.DataFrame([{
            "step": 1,
            "type": type_num,
            "amount": amount,
            "oldbalanceOrg": old_balance,
            "newbalanceOrig": new_balance,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0
        }])

        # Predict using trained model
        prediction = model.predict(input_data)[0]

        # Output result
        if prediction == 1:
            st.error("🚨 This transaction is FRAUDULENT.")
        else:
            st.success("✅ This transaction is LEGITIMATE.")
   
# Log transaction for monitoring
st.session_state.predicted_transactions.append({
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Amount": amount,
    "Type": transaction_type,
    "Old Balance": old_balance,
    "New Balance": new_balance,
    "Prediction": "FRAUD" if prediction == 1 else "LEGIT"
})


    except Exception as e:
        st.warning("Something went wrong while predicting.")
        st.text(str(e))

# =============================
# 📤 Upload & Monitor Page
# =============================
elif selected == "📤 Upload & Monitor":
    st.title("📤 Monitored Transactions")

    if st.session_state.predicted_transactions:
        df = pd.DataFrame(st.session_state.predicted_transactions)
        st.dataframe(df)

        fraud_count = (df["Prediction"] == "FRAUD").sum()
        st.metric("🚨 Total Fraudulent Transactions", fraud_count)

        # Export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download Fraud Report", csv, "fraud_report.csv", "text/csv")
    else:
        st.info("No transactions predicted yet.")

# =============================
# 📊 Reports Page
# =============================
elif selected == "📊 Reports":
    st.title("📊 Model Feature Importance")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        booster = model.get_booster()
        feature_names = booster.feature_names
        feat_imp = pd.Series(model.feature_importances_, index=feature_names)

        fig, ax = plt.subplots()
        feat_imp.sort_values().plot(kind='barh', ax=ax, color='purple')
        ax.set_title("Feature Importance", fontsize=14)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to load feature importance chart.\n\n{e}")





