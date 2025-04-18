import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# -------------------------------------
# PAGE CONFIGURATION
# -------------------------------------
st.set_page_config(page_title="AI Fraud Detection – Kareem Morad", layout="wide")
st.title("💼 AI-Powered Fraud Detection Dashboard")

# -------------------------------------
# DARK MODE & CUSTOM BACKGROUND
# -------------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #0e1117 !important;
            color: #ffffff !important;
            background-image: none !important;
        }
        .stTextInput, .stNumberInput, .stSelectbox, .stDataFrame, .stTextArea {
            color: white !important;
        }
        .stMarkdown, div, span, section {
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-image: url("https://images.unsplash.com/photo-1581092334783-f4476c2145ab?fit=crop&w=1600&q=80");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            scroll-behavior: smooth;
        }
        input:hover, select:hover, textarea:hover {
            background-color: #f0f8ff !important;
            transition: 0.3s ease;
        }
        .stButton>button {
            color: white !important;
            background-color: #2b7de9 !important;
            border-radius: 6px !important;
            transition: background-color 0.3s ease !important;
        }
        .stButton>button:hover {
            background-color: #1e63c4 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# -------------------------------------
# UPLOAD MODEL (.pkl)
# -------------------------------------
uploaded_model = st.sidebar.file_uploader("📤 Upload Trained Model (.pkl)", type="pkl")
model = None
if uploaded_model:
    with open("xgb_model.pkl", "wb") as f:
        f.write(uploaded_model.read())
    model = joblib.load("xgb_model.pkl")
    st.sidebar.success("✅ Model uploaded successfully.")

# -------------------------------------
# UPLOAD DATASET (.csv)
# -------------------------------------
uploaded_dataset = st.sidebar.file_uploader("📤 Upload Dataset (.csv)", type="csv")
if uploaded_dataset:
    with open("PS_20174392719_1491204439457_log.csv", "wb") as f:
        f.write(uploaded_dataset.read())
    st.sidebar.success("✅ Dataset uploaded successfully.")

# -------------------------------------
# SESSION STATE
# -------------------------------------
if "predicted_transactions" not in st.session_state:
    st.session_state.predicted_transactions = []

# -------------------------------------
# PDF REPORT GENERATOR
# -------------------------------------
def generate_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("Fraud Detection Report – Kareem Morad")

    pdf.cell(200, 10, txt="📄 Fraud Detection Report", ln=True, align='C')
    pdf.ln(10)

    total = len(df)
    frauds = df[df["Prediction"].str.contains("FRAUD")]
    fraud_count = len(frauds)

    pdf.cell(200, 10, txt=f"Total Transactions: {total}", ln=True)
    pdf.cell(200, 10, txt=f"Fraudulent Transactions: {fraud_count}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=10)

    for _, row in frauds.iterrows():
        line = f"{row['Timestamp']} | {row['Type']} | {row['Amount']} | FRAUD"
        pdf.cell(200, 8, txt=line, ln=True)

    pdf.cell(200, 10, txt="Generated by Kareem Morad", ln=True, align='C')
    pdf.output("fraud_report.pdf")

# -------------------------------------
# NAVIGATION
# -------------------------------------
section = st.sidebar.radio("Go to", ["🏠 Overview", "🔍 Predict", "📬 Upload & Monitor", "📁 All Logs", "📊 Reports"])

# -------------------------------------
# OVERVIEW
# -------------------------------------
if section == "🏠 Overview":
    st.subheader("📄 Project Overview")
    st.write("""
    This dashboard showcases an AI-powered fraud detection model built and trained by Kareem Morad.

    💡 Features:
    - Real-time transaction classification
    - Uploadable Model (.pkl) & Dataset (.csv)
    - Exportable fraud reports (PDF + CSV)
    - Permanent log saving
    - Full fraud analytics and performance reports
    """)

# -------------------------------------
# PREDICT
# -------------------------------------
elif section == "🔍 Predict":
    st.subheader("⚡ Real-Time Transaction Prediction")
    if model:
        amount = st.number_input("Transaction Amount", value=5000.0)
        tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        old_balance = st.number_input("Old Balance", value=10000.0)
        new_balance = st.number_input("New Balance", value=500.0)

        if st.button("🔍 Predict"):
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

                record = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Amount": amount,
                    "Type": tx_type,
                    "Old Balance": old_balance,
                    "New Balance": new_balance,
                    "Prediction": result
                }
                st.session_state.predicted_transactions.append(record)

                try:
                    log_df = pd.DataFrame([record])
                    log_df.to_csv("permanent_log.csv", mode='a', header=not os.path.exists("permanent_log.csv"), index=False)
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("⚠️ Please upload a trained model to start predictions.")

# -------------------------------------
# MONITOR TAB
# -------------------------------------
elif section == "📬 Upload & Monitor":
    st.subheader("📬 Monitored Transactions")
    if st.session_state.predicted_transactions:
        df_logs = pd.DataFrame(st.session_state.predicted_transactions)
        st.dataframe(df_logs)

        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Report", csv, "fraud_report.csv", "text/csv")

        if st.button("📄 Generate PDF Fraud Report"):
            generate_pdf_report(df_logs)
            with open("fraud_report.pdf", "rb") as file:
                st.download_button("⬇️ Download PDF", file.read(), file_name="fraud_report.pdf", mime="application/pdf")
    else:
        st.info("No transactions have been predicted yet.")

# -------------------------------------
# PERMANENT LOGS
# -------------------------------------
elif section == "📁 All Logs":
    st.subheader("📁 Permanent Log – All Transactions")
    try:
        full_logs = pd.read_csv("permanent_log.csv")
        st.dataframe(full_logs)
        st.download_button("⬇️ Download All Logs", full_logs.to_csv(index=False), file_name="permanent_log.csv", mime="text/csv")
    except FileNotFoundError:
        st.info("No permanent log file found yet.")

# -------------------------------------
# REPORTS
# -------------------------------------
elif section == "📊 Reports":
    st.subheader("📊 Model Evaluation & Fraud Insights")
    if model and os.path.exists("PS_20174392719_1491204439457_log.csv"):
        try:
            df = pd.read_csv("PS_20174392719_1491204439457_log.csv")
            df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
            df["type"] = df["type"].map({"TRANSFER": 0, "CASH_OUT": 1})
            df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"])

            features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            X = df[features]
            y = df["isFraud"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)

            st.markdown("### 📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
            st.pyplot(fig)

            st.markdown("### 📄 Classification Report")
            report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(report_df)

            st.markdown("### 🔥 Feature Correlation")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
            st.pyplot(fig2)

            st.markdown("### 📈 Transaction Amount Distribution")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.histplot(df[df["isFraud"] == 0]["amount"], bins=60, color="green", label="Legit", ax=ax3)
            sns.histplot(df[df["isFraud"] == 1]["amount"], bins=60, color="red", label="Fraud", ax=ax3)
            ax3.set_xlim(0, 200000)
            ax3.legend()
            ax3.set_title("Fraud vs Legit Amount Distribution")
            st.pyplot(fig3)

        except Exception as e:
            st.warning("⚠️ Unable to generate analytics.")
            st.text(str(e))
    else:
        st.error("❌ Model or dataset not found. Upload `.pkl` and `.csv` files to view analytics.")


