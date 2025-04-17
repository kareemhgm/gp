import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Simulated prediction logic (replace with model.predict() later)
def predict_fraud(amount, trans_type, old_balance, new_balance):
    if trans_type == "TRANSFER" and amount > 10000 and new_balance < old_balance:
        return 1
    return 0

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Fraud Detection", layout="wide")
st.title("🚨 AI-Powered Fraud Detection Dashboard")
st.markdown("**Developed by Kareem Morad  |  Supervised by Dr. Haitham Ghalwash**")

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("📁 Dashboard Menu")
page = st.sidebar.selectbox("Go to", ["Overview", "Monitoring", "Reports"])

# ------------------- SAMPLE DATA -------------------
sample_data = pd.DataFrame({
    "TransactionID": [1001, 1002, 1003, 1004, 1005],
    "Type": ["TRANSFER", "CASH_OUT", "CASH_OUT", "TRANSFER", "PAYMENT"],
    "Amount": [1000, 5000, 25000, 100000, 1500],
    "OldBalance": [5000, 10000, 60000, 120000, 2000],
    "NewBalance": [4000, 5000, 35000, 20000, 500],
    "IsFraud": [0, 1, 1, 0, 0]
})

# ------------------- OVERVIEW -------------------
if page == "Overview":
    st.subheader("📊 Model Overview")
    st.markdown("""
    This AI system analyzes transaction behavior to identify fraudulent activities.  
    The model was trained on over 500,000 mobile money transactions and selected based on high fraud detection precision and recall.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy", "100%")
    col2.metric("✅ Precision (Fraud)", "96%")
    col3.metric("🚨 Recall (Fraud)", "85%")

    st.markdown("### 📉 Confusion Matrix")
    cm = [[552384, 55], [240, 1403]]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# ------------------- MONITORING -------------------
elif page == "Monitoring":
    st.subheader("📡 Payment Monitoring")
    st.markdown("View incoming transactions and highlight potential fraud in real time.")
    
    fraud_filter = st.checkbox("🔍 Show only fraudulent transactions")
    if fraud_filter:
        st.dataframe(sample_data[sample_data["IsFraud"] == 1])
    else:
        st.dataframe(sample_data)

    st.markdown("---")
    st.subheader("🔎 Real-Time Transaction Prediction")

    amount = st.number_input("Transaction Amount", min_value=0)
    trans_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT"])
    old_balance = st.number_input("Old Balance", min_value=0)
    new_balance = st.number_input("New Balance", min_value=0)

    if st.button("Predict"):
        result = predict_fraud(amount, trans_type, old_balance, new_balance)
        if result == 1:
            st.error("⚠️ This transaction is likely FRAUDULENT.")
        else:
            st.success("✅ This transaction is LEGITIMATE.")

# ------------------- REPORTS -------------------
elif page == "Reports":
    st.subheader("📝 Fraud Detection Report")

    st.markdown("Download a CSV report of all flagged fraud transactions.")
    fraud_report = sample_data[sample_data["IsFraud"] == 1]
    st.dataframe(fraud_report)

    csv = fraud_report.to_csv(index=False)
    st.download_button(
        label="📥 Download Fraud Report",
        data=csv,
        file_name='fraud_report.csv',
        mime='text/csv'
    )

