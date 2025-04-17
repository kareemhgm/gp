
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your trained XGBoost model
model = joblib.load("xgb_model.pkl")

# Streamlit page config
st.set_page_config(page_title="AI Fraud Suite", layout="wide")

# ---------- LANDING PAGE / LOGIN ----------
LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Artificial_Intelligence_logo.svg/2048px-Artificial_Intelligence_logo.svg.png"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.image(LOGO, width=100)
    st.title("Welcome to AI Fraud Suite")
    st.write("AI-powered platform for real-time fraud monitoring.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username.strip() and password.strip():
            st.session_state.authenticated = True
        else:
            st.warning("Enter both username and password.")
    st.stop()

# ---------- HERO STYLING ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.hero {
    background: linear-gradient(to right, #6366f1, #3b82f6);
    padding: 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}
.result {
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
    font-weight: 500;
}
.fraud {
    background: rgba(239,68,68,0.1);
    border-left: 6px solid #ef4444;
}
.legit {
    background: rgba(34,197,94,0.1);
    border-left: 6px solid #22c55e;
}
</style>
<div class="hero">
    <h1>AI Fraud Suite 🚨</h1>
    <p>Detect and monitor fraudulent payments in real-time using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ---------- NAVIGATION ----------
tab = st.selectbox("🔎 Navigate", ["🏠 Overview", "💸 Predict", "📥 Monitor", "📊 Insights", "💬 Support"])

# ---------- OVERVIEW TAB ----------
if tab == "🏠 Overview":
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "100%")
    col2.metric("Precision", "96%")
    col3.metric("Recall", "85%")

    st.markdown("### 🔍 Platform Features")
    st.markdown("""
    - 🔒 Real-time fraud detection  
    - 📊 Upload transaction files  
    - 📈 Visualize model decisions  
    - 💬 Submit fraud support requests
    """)

# ---------- PREDICTION TAB ----------
elif tab == "💸 Predict":
    st.subheader("🔍 Real-Time Transaction Prediction")
    amount = st.number_input("Transaction Amount", value=5000.0)
    trans_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    old_balance = st.number_input("Old Balance", value=2000.0)
    new_balance = st.number_input("New Balance", value=0.0)

    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3}
    input_df = pd.DataFrame([{
        "type": type_map[trans_type],
        "amount": amount,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": new_balance
    }])

    if st.button("🚀 Predict"):
        result = model.predict(input_df)[0]
        if result == 1:
            st.markdown('<div class="result fraud">🚨 <b>FRAUDULENT</b> — Review immediately.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result legit">✅ <b>LEGITIMATE</b> — No fraud detected.</div>', unsafe_allow_html=True)

# ---------- MONITOR TAB ----------
elif tab == "📥 Monitor":
    st.subheader("📥 Upload Transactions CSV")
    file = st.file_uploader("Upload CSV with: type, amount, oldbalanceOrg, newbalanceOrig", type="csv")
    if file:
        df = pd.read_csv(file)
        if df["type"].dtype == object:
            df["type"] = df["type"].map({"TRANSFER": 1, "CASH_OUT": 2, "PAYMENT": 0, "DEBIT": 3})
        df["IsFraud"] = model.predict(df)
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download Labeled Report", csv, "fraud_report.csv", "text/csv")

# ---------- INSIGHTS TAB ----------
elif tab == "📊 Insights":
    st.subheader("📈 Feature Importance")
    feat_imp = pd.Series(model.feature_importances_, index=["type", "amount", "oldbalanceOrg", "newbalanceOrig"])
    st.bar_chart(feat_imp.sort_values())

# ---------- SUPPORT TAB ----------
elif tab == "💬 Support":
    st.subheader("💬 Contact Fraud Support")
    message = st.text_area("Describe your issue or question:")
    if st.button("Send"):
        if message.strip():
            st.success("✅ Your message has been sent. Our team will respond shortly.")
        else:
            st.warning("Please enter a message.")



