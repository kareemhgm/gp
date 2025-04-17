
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="wide")

st.title("🚨 AI-Powered Fraud Detection Dashboard")
st.subheader("Graduation Project by Kareem Morad")
st.markdown("**Supervised by: Dr. Haitham Ghalwash**")

st.markdown("### 📊 Model Summary")
st.markdown("""
- **Final Model:** XGBoost  
- **Trained on:** Full Byoot dataset  
- **Accuracy:** 100%  
- **Recall (Fraud):** 85%  
- **Precision (Fraud):** 96%  
- **F1-score (Fraud):** 90%
""")

st.markdown("### 📌 Confusion Matrix")
conf_matrix = [[552384, 55], [240, 1403]]
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.markdown("### 📈 Model Comparison Table")
comparison_data = {
    "Model": ["XGBoost", "Random Forest", "Logistic Regression", "KNN"],
    "Precision": [0.96, 1.00, 0.75, 1.00],
    "Recall": [0.85, 0.80, 0.60, 0.40],
    "F1-Score": [0.90, 0.89, 0.67, 0.57]
}
df = pd.DataFrame(comparison_data)
st.dataframe(df)

st.markdown("### 🧠 Research Notes")
st.markdown("""
- We tested 4 ML models: Random Forest, Logistic Regression, KNN, and XGBoost.  
- SMOTE was applied and evaluated for recall boost, but discarded due to low precision.  
- XGBoost achieved the best balance and was selected as the final model.  
- This system is designed for integration into **Byoot's payment infrastructure**.
""")

st.markdown("✅ This dashboard is part of the graduation project for real-time fraud detection monitoring.")
