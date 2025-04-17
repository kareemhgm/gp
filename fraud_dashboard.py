import streamlit as st
import pandas as pd
import joblib

# Load your trained XGBoost model (make sure it's in the same repo)
xgb_model = joblib.load('xgb_model.pkl')

# Page settings
st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# ---------- UI STYLING ----------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #F9FAFB;
    }
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 2.8rem;
        font-weight: 600;
        color: #111827;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: -1rem;
        padding-bottom: 1rem;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
    }
    .metric-box {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #FFFFFF;
        border-left: 6px solid #3B82F6;
        padding: 1rem;
        margin-top: 1.5rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07);
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">💳 AI-Powered Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect and monitor fraudulent payment transactions using machine learning.</div>', unsafe_allow_html=True)

# ---------- USER INPUT ----------
st.subheader("🔍 Real-Time Transaction Prediction")

col1, col2 = st.columns(2)
amount = col1.number_input("Transaction Amount", min_value=0.0, step=100.0, value=5000.0)
type_input = col2.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])

col3, col4 = st.columns(2)
old_balance = col3.number_input("Old Balance", min_value=0.0, step=50.0, value=3000.0)
new_balance = col4.number_input("New Balance", min_value=0.0, step=50.0, value=0.0)

# ---------- PREDICTION ----------
type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3}
type_encoded = type_map[type_input]

if st.button("🧠 Predict"):
    input_data = pd.DataFrame([{
        'type': type_encoded,
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance
    }])

    prediction = xgb_model.predict(input_data)[0]

    if prediction == 1:
        st.markdown('<div class="result-card" style="border-left-color:#EF4444">🚨 This transaction is <b>FRAUDULENT</b>.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card" style="border-left-color:#10B981">✅ This transaction is <b>LEGITIMATE</b>.</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("© 2025 AI Fraud Intelligence | Built with Streamlit + XGBoost")


