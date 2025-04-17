import streamlit as st
import pandas as pd
import joblib

# Load model (make sure xgb_model.pkl is in your repo)
model = joblib.load("xgb_model.pkl")

# Page config
st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# Custom CSS for modern SaaS design
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #f0f4f8, #f9fafb);
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(to right, #3b82f6, #9333ea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            font-size: 1.2rem;
            color: #6b7280;
            margin-top: -1rem;
            padding-bottom: 1rem;
        }
        .glass {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            margin-top: 2rem;
        }
        .stButton>button {
            background: linear-gradient(to right, #6366f1, #3b82f6);
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.5rem;
            border-radius: 10px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #4f46e5, #2563eb);
            transform: scale(1.03);
        }
        .result-good {
            background: rgba(34,197,94,0.1);
            padding: 1rem;
            border-radius: 12px;
            border-left: 6px solid #22c55e;
            margin-top: 2rem;
        }
        .result-bad {
            background: rgba(239,68,68,0.1);
            padding: 1rem;
            border-radius: 12px;
            border-left: 6px solid #ef4444;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Hero Section ----------
st.markdown('<div class="hero-title">AI-Powered Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Secure your transactions with real-time machine learning insights.</div>', unsafe_allow_html=True)

# ---------- Prediction Form ----------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🔍 Enter Transaction Details")

col1, col2 = st.columns(2)
amount = col1.number_input("Transaction Amount", min_value=0.0, value=5000.0)
type_input = col2.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])

col3, col4 = st.columns(2)
old_balance = col3.number_input("Old Balance", min_value=0.0, value=2000.0)
new_balance = col4.number_input("New Balance", min_value=0.0, value=0.0)

# Prepare input
type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3}
input_df = pd.DataFrame([{
    "type": type_map[type_input],
    "amount": amount,
    "oldbalanceOrg": old_balance,
    "newbalanceOrig": new_balance
}])

# ---------- Predict ----------
if st.button("🚀 Predict Fraud"):
    result = model.predict(input_df)[0]
    if result == 1:
        st.markdown('<div class="result-bad"><h4>🚨 FRAUDULENT</h4>This transaction is likely fraudulent. Immediate review recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-good"><h4>✅ LEGITIMATE</h4>No fraud detected in this transaction.</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("Designed by Kareem | © 2025 AI Fraud Suite | Inspired by Sift.com")




