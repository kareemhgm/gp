import streamlit as st
import pandas as pd
import joblib

# Load model (ensure xgb_model.pkl is in your repo)
model = joblib.load("xgb_model.pkl")

# Page config
st.set_page_config(page_title="Fraud Detection SaaS", layout="wide")

# Custom CSS for dashboard feel
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.metric-card {
    padding: 1rem;
    border-radius: 10px;
    background-color: #F3F4F6;
    text-align: center;
    margin-bottom: 1rem;
}
.card-pass {
    border-left: 6px solid #10B981;
}
.card-fail {
    border-left: 6px solid #EF4444;
}
</style>
""", unsafe_allow_html=True)

# Navigation Sidebar
tab = st.sidebar.radio("📁 Navigation", ["Overview", "Prediction", "Monitoring", "Reports"])

# ---------------- OVERVIEW ----------------
if tab == "Overview":
    st.title("📊 Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card card-pass"><h3>Accuracy</h3><h1>100%</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card card-pass"><h3>Precision</h3><h1>96%</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card card-pass"><h3>Recall</h3><h1>85%</h1></div>', unsafe_allow_html=True)
    st.write("This dashboard is powered by an XGBoost model trained on 500,000+ transactions from the PaySim dataset.")

# ---------------- PREDICTION ----------------
elif tab == "Prediction":
    st.title("🧠 Real-Time Fraud Prediction")
    
    col1, col2 = st.columns(2)
    amount = col1.number_input("Transaction Amount", value=5000.0)
    trans_type = col2.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    
    col3, col4 = st.columns(2)
    old_balance = col3.number_input("Old Balance", value=2000.0)
    new_balance = col4.number_input("New Balance", value=0.0)

    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3}
    input_data = pd.DataFrame([{
        'type': type_map[trans_type],
        'amount': amount,
        'oldbalanceOrg': old_balance,
        'newbalanceOrig': new_balance
    }])

    if st.button("🔍 Predict Fraud"):
        pred = model.predict(input_data)[0]
        if pred == 1:
            st.markdown('<div class="metric-card card-fail"><h2>🚨 FRAUDULENT Transaction</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card card-pass"><h2>✅ Legitimate Transaction</h2></div>', unsafe_allow_html=True)

# ---------------- MONITORING ----------------
elif tab == "Monitoring":
    st.title("📡 Live Transaction Monitoring")
    st.info("Live data stream and fraud flags will be displayed here soon.")

# ---------------- REPORTS ----------------
elif tab == "Reports":
    st.title("📄 Fraud Detection Reports")
    st.warning("Report export and summary features will be added in upcoming updates.")



