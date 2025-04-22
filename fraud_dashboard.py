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
import zipfile
import gdown

# -------------------------------------
# CONFIGURATION
# -------------------------------------
st.set_page_config(page_title="AI Fraud Detection – Kareem Morad", layout="wide")

# -------------------------------------
# STYLING
# -------------------------------------
st.markdown("""
    <style>
    body {
        background: radial-gradient(circle at top left, #0a0f1f, #000000) no-repeat center center fixed;
        background-size: cover;
        color: white;
    }
    h1, h2, h3, .stMarkdown, .stTitle {
        color: #6cc3ff !important;
        text-shadow: 0 0 10px #6cc3ff;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        color: white;
    }
    .stButton>button {
        background-color: #0f62fe;
        color: white;
        border: none;
        border-radius: 8px;
        box-shadow: 0 0 8px #6cc3ff;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0043ce;
        transform: scale(1.02);
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stTextArea {
        background-color: #1c1f26 !important;
        color: white !important;
        border-radius: 6px !important;
    }
    footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------
# MODEL UPLOAD
# -------------------------------------
uploaded_model = st.sidebar.file_uploader("📤 Upload Trained Model (.pkl)", type="pkl")
model = None
if uploaded_model:
    with open("xgb_model.pkl", "wb") as f:
        f.write(uploaded_model.read())
    model = joblib.load("xgb_model.pkl")
    st.session_state["model"] = model
    st.sidebar.success("✅ Model uploaded successfully.")

# -------------------------------------
# LOAD AND UNZIP CSV FROM GOOGLE DRIVE
# -------------------------------------
csv_url = st.sidebar.text_input("📥 Paste Google Drive Share Link")
csv_ready = False

if csv_url and "drive.google.com" in csv_url:
    try:
        file_id = csv_url.split("/d/")[1].split("/")[0]
        output_zip_path = "dataset.zip"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip_path, quiet=False)

        with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
            zip_ref.extractall("unzipped_data")

        for file in os.listdir("unzipped_data"):
            if file.endswith(".csv"):
                csv_path = os.path.join("unzipped_data", file)
                st.session_state["uploaded_csv_path"] = csv_path
                csv_ready = True
                st.sidebar.success(f"✅ Dataset extracted: {file}")
                break
        else:
            st.sidebar.error("❌ No CSV file found in ZIP.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to download or unzip: {e}")

# -------------------------------------
# SESSION INITIALIZATION
# -------------------------------------
if "predicted_transactions" not in st.session_state:
    st.session_state.predicted_transactions = []

# -------------------------------------
# PDF REPORT FUNCTION
# -------------------------------------
def generate_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("Fraud Detection Report – Kareem Morad")
    pdf.cell(200, 10, txt="Fraud Detection Report", ln=True, align='C')
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
        pdf.cell(200, 8, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)

    pdf.output("fraud_report.pdf")

# -------------------------------------
# NAVIGATION
# -------------------------------------
section = st.sidebar.radio("Go to", ["🏠 Overview", "🔍 Predict", "📬 Upload & Monitor", "📁 All Logs", "📊 Reports"])

# -------------------------------------
# SECTIONS
# -------------------------------------
if section == "🏠 Overview":
    st.subheader("📄 Project Overview")
    st.write("""
    This dashboard showcases an AI-powered fraud detection model built and trained by Kareem Morad.
    - Real-time transaction classification
    - Uploadable Model (.pkl) & Dataset (.csv)
    - Exportable fraud reports
    - Full fraud analytics
    """)

elif section == "🔍 Predict":
    st.subheader("⚡ Real-Time Transaction Prediction")
    if model:
        amount = st.number_input("Transaction Amount", value=5000.0)
        tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        old_balance = st.number_input("Old Balance (Origin)", value=10000.0)
        new_balance = st.number_input("New Balance (Origin)", value=500.0)

        diff_orig = old_balance - amount
        est_new_dest = new_balance + amount
        flag_old_zero = int(old_balance == 0)
        flag_new_zero = int(new_balance == 0)

        if st.button("🔍 Predict"):
            try:
                type_map = {"TRANSFER": 0, "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "DEBIT": 4}
                input_data = pd.DataFrame([[
                    type_map[tx_type], amount, old_balance, new_balance,
                    diff_orig, est_new_dest, flag_old_zero, flag_new_zero
                ]], columns=[
                    'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'diffOrig', 'estNewDest', 'flagOldZero', 'flagNewZero'
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
                pd.DataFrame([record]).to_csv("permanent_log.csv", mode='a', header=not os.path.exists("permanent_log.csv"), index=False)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("⚠️ Please upload a trained model to start predictions.")

elif section == "📬 Upload & Monitor":
    st.subheader("📬 Monitored Transactions")
    if st.session_state.predicted_transactions:
        df_logs = pd.DataFrame(st.session_state.predicted_transactions)
        st.dataframe(df_logs)
        st.download_button("⬇️ Download CSV", df_logs.to_csv(index=False), "fraud_report.csv")
        if st.button("📄 Generate PDF"):
            generate_pdf_report(df_logs)
            with open("fraud_report.pdf", "rb") as file:
                st.download_button("⬇️ Download PDF", file.read(), "fraud_report.pdf")
    else:
        st.info("No transactions yet.")

elif section == "📁 All Logs":
    st.subheader("📁 Permanent Log – All Transactions")
    try:
        logs = pd.read_csv("permanent_log.csv")
        st.dataframe(logs)
        st.download_button("⬇️ Download Log", logs.to_csv(index=False), "permanent_log.csv")
    except FileNotFoundError:
        st.info("No logs yet.")

elif section == "📊 Reports":
    st.subheader("📊 Model Evaluation & Fraud Insights")
    if not model or "uploaded_csv_path" not in st.session_state:
        st.error("❌ Model or dataset not found.")
        st.stop()

    try:
        df = pd.read_csv(st.session_state["uploaded_csv_path"])
        if "type" in df.columns:
            df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
            df["type"] = df["type"].map({"TRANSFER": 0, "CASH_OUT": 1})
        drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "step"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

        df["diffOrig"] = df["oldbalanceOrg"] - df["amount"]
        df["estNewDest"] = df["newbalanceOrig"] + df["amount"]
        df["flagOldZero"] = (df["oldbalanceOrg"] == 0).astype(int)
        df["flagNewZero"] = (df["newbalanceOrig"] == 0).astype(int)

        if not all(col in df.columns for col in ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'diffOrig', 'estNewDest', 'flagOldZero', 'flagNewZero', 'isFraud']):
            st.warning("⚠️ Missing required columns.")
            st.stop()

        features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'diffOrig', 'estNewDest', 'flagOldZero', 'flagNewZero']
        X = df[features]
        y = df["isFraud"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        st.markdown("### 📌 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        st.pyplot(fig)

        st.markdown("### 📄 Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

        st.markdown("### 🔥 Feature Correlation")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    except Exception as e:
        st.warning("⚠️ Could not generate analytics.")
        st.text(str(e))



