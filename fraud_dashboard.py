import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
model = joblib.load('xgb_fraud_model.pkl')

# Streamlit config
st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="wide")
st.title("🔍 Real-Time Transaction Fraud Detection")

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown("<style>body { background-color: #0e1117; color: #fafafa; }</style>", unsafe_allow_html=True)

# Session state for logging
if 'predicted_transactions' not in st.session_state:
    st.session_state.predicted_transactions = []

# Sidebar navigation
st.sidebar.markdown("📑 **Navigation**")
section = st.sidebar.radio("Go to", ["🏠 Overview", "🔍 Predict", "📬 Upload & Monitor", "📊 Reports"])

# ------------------------- Overview -------------------------
if section == "🏠 Overview":
    st.subheader("📄 Project Overview")
    st.write("""
        Welcome to the AI-Powered Fraud Detection Dashboard.
        This platform allows you to:
        - Test real-time transactions and detect fraud
        - Monitor predictions and export reports
        - Visualize insights from the fraud detection model
    """)

# ------------------------- Predict -------------------------
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
            st.error(f"Error during prediction: {e}")

# ------------------------- Monitor -------------------------
elif section == "📬 Upload & Monitor":
    st.subheader("📬 Monitored Transactions")

    if st.session_state.predicted_transactions:
        df_logs = pd.DataFrame(st.session_state.predicted_transactions)
        st.dataframe(df_logs)

        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Report", csv, "fraud_report.csv", "text/csv")
    else:
        st.info("No transactions monitored yet.")

# ------------------------- Reports -------------------------
elif section == "📊 Reports":
    st.subheader("📊 Fraud Detection Insights & Model Evaluation")

    try:
        df_full = pd.read_csv("PS_20174392719_1491204439457_log.csv")
        df_full = df_full[df_full["type"].isin(["TRANSFER", "CASH_OUT"])]
        df_full["type"] = df_full["type"].map({"TRANSFER": 0, "CASH_OUT": 1})
        df_full = df_full.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"])

        features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        X = df_full[features]
        y = df_full['isFraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # === Model Testing ===
        st.markdown("### 🧠 Model Evaluation")

        y_pred = model.predict(X_test)

        st.markdown("#### 📌 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        st.pyplot(fig)

        st.markdown("#### 📄 Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df)

        # === Correlation Heatmap ===
        st.markdown("### 🔥 Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_full.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)

        # === Amount Distribution ===
        st.markdown("### 📈 Transaction Amount Distribution")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df_full[df_full['isFraud'] == 0], x='amount', bins=60, color='green', label='Legit', ax=ax3)
        sns.histplot(data=df_full[df_full['isFraud'] == 1], x='amount', bins=60, color='red', label='Fraud', ax=ax3)
        ax3.set_xlim(0, 200000)
        ax3.set_title("Fraud vs Legit by Transaction Amount")
        ax3.legend()
        st.pyplot(fig3)

        # === Balance Difference Boxplot ===
        st.markdown("### 📉 Balance Difference in Fraud Cases")
        df_full["balanceDiff"] = df_full["oldbalanceOrg"] - df_full["newbalanceOrig"]
        fig4, ax4 = plt.subplots()
        sns.boxplot(x="isFraud", y="balanceDiff", data=df_full, palette=["green", "red"], ax=ax4)
        ax4.set_title("Balance Drop in Fraud vs Legit")
        st.pyplot(fig4)

    except Exception as e:
        st.warning("⚠️ Could not load dataset or generate analytics.")
        st.text(str(e))






