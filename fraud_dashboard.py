import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your trained XGBoost model
model = joblib.load("xgb_model.pkl")

# Setup page
st.set_page_config(page_title="AI Fraud Suite", layout="wide")

# 🌙 Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# Dynamic style
if dark_mode:
    background = "#111827"
    font_color = "#F9FAFB"
    card_color = "#1F2937"
    good_color = "#059669"
    bad_color = "#DC2626"
else:
    background = "#F9FAFB"
    font_color = "#111827"
    card_color = "#FFFFFF"
    good_color = "#16A34A"
    bad_color = "#DC2626"

# Custom styling
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    background-color: {background};
    color: {font_color};
    font-family: 'Segoe UI', sans-serif;
}}
.stButton>button {{
    background: linear-gradient(to right, #6366f1, #3b82f6);
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    border-radius: 10px;
    transition: 0.3s ease;
}}
.stButton>button:hover {{
    background: linear-gradient(to right, #4f46e5, #2563eb);
    transform: scale(1.03);
}}
.result-good {{
    background: rgba(34,197,94,0.1);
    padding: 1rem;
    border-radius: 12px;
    border-left: 6px solid {good_color};
    margin-top: 2rem;
}}
.result-bad {{
    background: rgba(239,68,68,0.1);
    padding: 1rem;
    border-radius: 12px;
    border-left: 6px solid {bad_color};
    margin-top: 2rem;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------
tab = st.sidebar.radio("📁 Navigation", ["🏠 Overview", "🔍 Predict", "📥 Upload & Monitor", "📈 Reports"])

# ---------------- OVERVIEW ----------------
if tab == "🏠 Overview":
    st.title("AI-Powered Fraud Detection")
    st.write("Built with XGBoost and PaySim data for real-time payment fraud detection.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "100%")
    col2.metric("Precision", "96%")
    col3.metric("Recall", "85%")
    st.markdown("This dashboard allows for real-time fraud detection, CSV-based transaction analysis, and model transparency via feature importance.")

# ---------------- REAL-TIME PREDICTION ----------------
elif tab == "🔍 Predict":
    st.subheader("🔍 Real-Time Transaction Prediction")
    col1, col2 = st.columns(2)
    amount = col1.number_input("Transaction Amount", value=5000.0)
    trans_type = col2.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    col3, col4 = st.columns(2)
    old_balance = col3.number_input("Old Balance", value=2000.0)
    new_balance = col4.number_input("New Balance", value=0.0)

    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3}
    input_df = pd.DataFrame([{
        "type": type_map[trans_type],
        "amount": amount,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": new_balance
    }])

    if st.button("🚀 Predict Fraud"):
        result = model.predict(input_df)[0]
        if result == 1:
            st.markdown('<div class="result-bad"><h4>🚨 FRAUDULENT</h4>This transaction is likely fraudulent.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-good"><h4>✅ LEGITIMATE</h4>No fraud detected in this transaction.</div>', unsafe_allow_html=True)

# ---------------- CSV UPLOAD + MONITOR ----------------
elif tab == "📥 Upload & Monitor":
    st.subheader("📥 Upload a Transaction CSV File")
    uploaded_file = st.file_uploader("Upload CSV with: type, amount, oldbalanceOrg, newbalanceOrig", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Convert type if needed
        type_dict = {"TRANSFER": 1, "CASH_OUT": 2, "PAYMENT": 0, "DEBIT": 3}
        if df['type'].dtype == object:
            df['type'] = df['type'].map(type_dict)

        preds = model.predict(df)
        df["IsFraud"] = preds

        st.write("📊 Preview with Predictions:")
        st.dataframe(df.head())

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Labeled Report", csv, "fraud_report.csv", "text/csv")

# ---------------- REPORTS / FEATURE IMPORTANCE ----------------
elif tab == "📈 Reports":
    st.subheader("📈 Model Feature Importance")

    if hasattr(model, "feature_importances_"):
        feat_imp = pd.Series(model.feature_importances_, index=["type", "amount", "oldbalanceOrg", "newbalanceOrig"])
        st.bar_chart(feat_imp.sort_values())
    else:
        st.info("Model does not support feature_importances_")




