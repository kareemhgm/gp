import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="wide")
st.title("🚨 AI-Powered Fraud Detection Dashboard")
st.markdown("**By Kareem Morad | Supervised by Dr. Haitham Ghalwash**")

st.sidebar.title("📁 Dashboard Menu")
page = st.sidebar.selectbox("Go to", ["Overview", "Monitoring", "Reports"])

if page == "Overview":
    st.subheader("📊 Model Overview")
    st.markdown("""
    Welcome to the AI-Powered Fraud Detection system.  
    This dashboard provides insights into our machine learning model performance, fraud detection analysis, and real-time payment monitoring.
    """)

elif page == "Monitoring":
    st.subheader("📡 Payment Monitoring")
    st.info("Live transaction view will appear here...")

elif page == "Reports":
    st.subheader("📝 Fraud Detection Report")
    st.success("You’ll be able to download a report of flagged fraud transactions here.")

