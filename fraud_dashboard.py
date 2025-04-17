import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== STYLING ==========
st.markdown("""
    <style>
        body, .stApp {
            background-color: #f3f4f6;
            font-family: 'Segoe UI', sans-serif;
        }

        h1, h2, h3, h4 {
            color: #1f2937;
        }

        .stButton>button {
            background-color: #6366f1;
            color: white;
            border-radius: 6px;
            padding: 0.5em 1em;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #4f46e5;
            transform: scale(1.02);
        }

        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.title("🧠 AI-Powered Fraud Detection System")
st.markdown("A smart tool to detect fraudulent transactions in real time.")

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")  # Make sure model is saved as .pkl

model = load_model()

# ========== SIDEBAR ==========
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Overview", "🔍 Predict", "📤 Upload & Monitor", "📊 Reports"])

# ========== OVERVIEW PAGE ==========
if page == "🏠 Overview":
    st.header("📌 Project Summary")
    st.markdown("""
        This dashboard enables you to:
        - Analyze and monitor real-time financial transactions
        - Predict fraudulent behavior using an XGBoost AI model
        - Upload datasets and evaluate risks interactively
        - Generate visual and tabular reports
    """)

# ========== PREDICT PAGE ==========
elif page == "🔍 Predict":
    st.header("🔎 Real-Time Transaction Prediction")

    amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    old_balance = st.number_input("Old Balance", min_value=0.0, step=0.01)
    new_balance = st.number_input("New Balance", min_value=0.0, step=0.01)

    if st.button("🔍 Predict"):
        type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
        input_data = pd.DataFrame([{
            "type": type_map[transaction_type],
            "amount": amount,
            "oldbalanceOrg": old_balance,
            "newbalanceOrig": new_balance
        }])

        try:
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ This transaction is LEGITIMATE.")
        except Exception as e:
            st.warning("Something went wrong while predicting.")
            st.text(str(e))

# ========== UPLOAD PAGE ==========
elif page == "📤 Upload & Monitor":
    st.header("📤 Upload Transactions Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("📈 Run Fraud Detection"):
            type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
            if 'type' in df.columns and df['type'].dtype == object:
                df['type'] = df['type'].map(type_map)

            input_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
            try:
                df['prediction'] = model.predict(df[input_cols])
                st.success("✅ Detection Complete. Here are results:")
                st.dataframe(df[['amount', 'type', 'prediction']].head())

                fraud_count = df['prediction'].sum()
                st.metric("🚨 Fraudulent Transactions", fraud_count)

                st.download_button("Download Results", df.to_csv(index=False), "fraud_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ========== REPORTS PAGE ==========
elif page == "📊 Reports":
    st.header("📊 Model Feature Importance")

    try:
        feature_names = model.get_booster().feature_names
        feat_imp = pd.Series(model.feature_importances_, index=feature_names)

        st.subheader("🔍 Feature Importances")
        st.bar_chart(feat_imp.sort_values(ascending=False))
    except Exception as e:
        st.warning("Feature importance not available.")
        st.text(str(e))





