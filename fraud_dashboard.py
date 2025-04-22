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
        background: radial-gradient(circle at top left, #0a0f1f, #000000);
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
        border-radius: 8px;
        box-shadow: 0 0 8px #6cc3ff;
    }
    .stButton>button:hover {
        background-color: #0043ce;
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
# REPORT SECTION
# -------------------------------------
st.title("📊 AI Fraud Detection Analysis")
if not model or "uploaded_csv_path" not in st.session_state:
    st.warning("Upload both a trained model and a zipped dataset to begin.")
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

    required_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'diffOrig',
                     'estNewDest', 'flagOldZero', 'flagNewZero', 'isFraud']
    if not all(col in df.columns for col in required_cols):
        st.warning("⚠️ Missing required columns.")
        st.stop()

    X = df[[col for col in required_cols if col != 'isFraud']]
    y = df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    st.markdown("### 📌 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
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






