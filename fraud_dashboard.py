import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

model = joblib.load('xgb_fraud_model.pkl')


# Set page config
st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="wide")

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("<style>body { background-color: #0e1117; color: #fafafa; }</style>", unsafe_allow_html=True)

# Initialize session state
if 'predicted_transactions' not in st.session_state:
    st.session_state.predicted_transactions = []

# Sidebar navigation
st.sidebar.markdown("📑 **Navigation**")
section = st.sidebar.radio("Go to", ["🏠 Overview", "🔍 Predict", "📬 Upload & Monitor", "📊 Reports"])

# Main App Interface
st.title("🔍 Real-Time Transaction Prediction")

if section == "🏠 Overview":
    st.subheader("📄 Project Overview")
    st.write("""
        This AI-powered dashboard uses a machine learning model trained on transaction data to detect fraudulent activities. 
        You can test real-time predictions, monitor transactions, and view fraud detection reports.
    """)

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
            color = "red" if prediction == 1 else "green"
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
            st.error(f"Something went wrong: {e}")

elif section == "📬 Upload & Monitor":
    st.subheader("📬 Monitored Transactions")
    st.write("Log of all tested transactions.")

    if st.session_state.predicted_transactions:
        df_logs = pd.DataFrame(st.session_state.predicted_transactions)
        st.dataframe(df_logs)
    else:
        st.info("No transactions monitored yet.")

elif section == "📊 Reports":
    elif section == "📊 Reports":
    st.subheader("📊 Fraud Insights & Analysis")
elif section == "📊 Reports":
    st.subheader("📊 Fraud Detection Insights")
    st.markdown("### 🧠 Model Evaluation")

    from sklearn.metrics import classification_report, confusion_matrix

    # Predict on test set
    y_pred = model.predict(X_test)

    # Show Confusion Matrix
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    st.pyplot(fig)

    # Show Classification Report
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


    try:
        # Load your dataset
        df_full = pd.read_csv("PS_20174392719_1491204439457_log.csv")

        # Keep only relevant types
        df_full = df_full[df_full["type"].isin(["TRANSFER", "CASH_OUT"])]
        df_full["type"] = df_full["type"].map({"TRANSFER": 0, "CASH_OUT": 1})

        # Drop unused columns
        df_full = df_full.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"])

        # 🔥 1. Correlation Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        import seaborn as sns
        import matplotlib.pyplot as plt

        corr = df_full.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # 📈 2. Amount Distribution by Class
        st.markdown("### 📈 Transaction Amount Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df_full[df_full['isFraud'] == 0], x='amount', bins=60, color='green', label='Legit', ax=ax2)
        sns.histplot(data=df_full[df_full['isFraud'] == 1], x='amount', bins=60, color='red', label='Fraud', ax=ax2)
        ax2.set_xlim(0, 200000)
        ax2.set_title("Transaction Amounts – Fraud vs Legit")
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.warning("⚠️ Could not load dataset for analytics.")
        st.text(str(e))

    # Load the dataset (optional: replace with your full cleaned dataset if needed)
    try:
        df_full = pd.read_csv("PS_20174392719_1491204439457_log.csv")

        # Filter to match model usage
        df_full = df_full[df_full["type"].isin(["TRANSFER", "CASH_OUT"])]
        df_full["type"] = df_full["type"].map({"TRANSFER": 0, "CASH_OUT": 1})
        df_full = df_full.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"])

        st.markdown("### 🔥 Correlation Heatmap")
        import seaborn as sns
        import matplotlib.pyplot as plt

        corr = df_full.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.markdown("### 📈 Distribution of Transaction Amounts")
        fig2, ax2 = plt.subplots()
        sns.histplot(df_full[df_full['isFraud'] == 0]['amount'], bins=50, color='green', label='Legit', ax=ax2)
        sns.histplot(df_full[df_full['isFraud'] == 1]['amount'], bins=50, color='red', label='Fraud', ax=ax2)
        ax2.set_title("Transaction Amount Distribution by Class")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("### 📉 Balance Differences in Fraud Cases")
        df_full["balanceDiff"] = df_full["oldbalanceOrg"] - df_full["newbalanceOrig"]
        fig3, ax3 = plt.subplots()
        sns.boxplot(x="isFraud", y="balanceDiff", data=df_full, palette=["green", "red"], ax=ax3)
        ax3.set_title("Balance Drop in Fraud vs Legit")
        st.pyplot(fig3)

    except Exception as e:
        st.warning("⚠️ Could not load dataset for report visuals.")
        st.text(str(e))

    st.subheader("📊 Model Performance Report")
    st.write("This section will soon include visualizations of performance metrics and insights from the full test set.")
    st.info("Coming soon!")






