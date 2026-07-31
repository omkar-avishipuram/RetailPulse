import streamlit as st
from pathlib import Path

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="RetailPulse Dashboard",
    page_icon="🛒",
    layout="wide",
)

# ----------------------------------------
# Header
# ----------------------------------------
st.title("🛒 RetailPulse")
st.subheader("Retail Analytics & Machine Learning Dashboard")

st.markdown(
    """
Welcome to **RetailPulse**, an end-to-end retail analytics project built with
**Python, Streamlit, Machine Learning, and Data Visualization**.

Use the navigation menu on the left to explore the project.
"""
)

# ----------------------------------------
# Project Overview
# ----------------------------------------
st.header("📌 Project Overview")

st.write(
    """
RetailPulse helps retailers analyze sales data, forecast demand,
segment customers, predict churn, and evaluate inventory performance.

The dashboard is divided into multiple modules covering the complete
data science workflow.
"""
)

# ----------------------------------------
# Dashboard Modules
# ----------------------------------------
st.header("📊 Dashboard Modules")

col1, col2 = st.columns(2)

with col1:
    st.info(
        """
### 📊 EDA
- Dataset exploration
- Data cleaning
- Missing values
- Sales trends
- Revenue analysis
- Customer insights
"""
    )

    st.info(
        """
### 📈 Forecasting
- Monthly sales
- Demand forecasting
- Trend visualization
- Revenue prediction
"""
    )

    st.info(
        """
### 👥 Customer Segmentation
- Feature engineering
- RFM analysis
- K-Means clustering
- Customer groups
"""
    )

with col2:
    st.info(
        """
### 🚪 Customer Churn
- Churn dataset
- Model training
- Performance metrics
- Predictions
"""
    )

    st.info(
        """
### 📦 Inventory
- Model evaluation
- Confusion matrix
- ROC curve
- Feature importance
"""
    )

    st.info(
        """
### 📁 Generated Files
- Clean dataset
- RFM dataset
- Customer segments
- Churn dataset
- Trained models
"""
    )

# ----------------------------------------
# Technologies Used
# ----------------------------------------
st.header("🛠 Technologies")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.markdown(
        """
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
"""
    )

with tech2:
    st.markdown(
        """
- Scikit-learn
- Streamlit
- Joblib
- OpenPyXL
"""
    )

with tech3:
    st.markdown(
        """
- Git
- GitHub
- VS Code
- Jupyter Notebook
"""
    )

# ----------------------------------------
# Project Structure
# ----------------------------------------
st.header("📂 Project Structure")

st.code(
"""
RetailPulse/
│
├── dashboard/
│   ├── Home.py
│   └── pages/
│       ├── 1_EDA.py
│       ├── 2_Forecasting.py
│       ├── 3_Segmentation.py
│       ├── 4_Churn.py
│       └── 5_Inventory.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── screenshots/
├── requirements.txt
└── README.md
""",
    language="text",
)

# ----------------------------------------
# Quick Statistics
# ----------------------------------------
st.header("📈 Dashboard Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Modules", "5")
c2.metric("ML Models", "3")
c3.metric("Datasets", "4")
c4.metric("Visualizations", "15+")
c5.metric("Framework", "Streamlit")

# ----------------------------------------
# Footer
# ----------------------------------------
st.divider()

st.success("✅ RetailPulse Dashboard loaded successfully!")

st.caption(
    "RetailPulse | End-to-End Retail Analytics using Streamlit & Machine Learning"
)