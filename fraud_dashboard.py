import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

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
elif selected == "🔍 Predict":
    st.title("🔍 Real-Time Transaction Prediction")

    amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
    type_input = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT"])
    old_balance = st.number_input("Old Balance", min_value=0.0, format="%.2f")
    new_balance = st.number_input("New Balance", min_value=0.0, format="%.2f")

    if st.button("🔮 Predict", use_container_width=True):
        try:
            input_df = pd.DataFrame([{
                'type': type_input,
                'amount': amount,
                'oldbalanceOrg': old_balance,
                'newbalanceOrig': new_balance,
                'oldbalanceDest': 0.0,
                'newbalanceDest': 0.0,
                'isFlaggedFraud': 0,
                'step': 1
            }])
            
            # Ensure correct order
            input_df = input_df[["type", "amount", "oldbalanceOrg", "newbalanceOrig", 
                                 "oldbalanceDest", "newbalanceDest", "isFlaggedFraud", "step"]]

            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.error("🚨 This transaction is FRAUDULENT.")
            else:
                st.success("✅ This transaction is LEGITIMATE.")
        except Exception as e:
            st.warning(f"Something went wrong while predicting.\n\n{e}")

# =============================
# 📤 Upload & Monitor Page
# =============================
elif selected == "📤 Upload & Monitor":
    st.title("📤 Upload CSV for Bulk Analysis")

    uploaded_file = st.file_uploader("Upload your transaction dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("🧾 Preview of Uploaded Transactions:")
        st.dataframe(df.head())

        # Make predictions if structure is correct
        try:
            df = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig", 
                     "oldbalanceDest", "newbalanceDest", "isFlaggedFraud", "step"]]
            df["Prediction"] = model.predict(df)
            frauds = df[df["Prediction"] == 1]
            st.success(f"✅ Scanned {len(df)} transactions. Found {len(frauds)} potential fraud cases.")
            st.dataframe(frauds)
        except Exception as e:
            st.warning(f"⚠️ Could not predict due to incorrect file format.\n\n{e}")

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





