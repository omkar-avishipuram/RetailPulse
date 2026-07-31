import streamlit as st

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
st.caption("Retail Analytics & Machine Learning Dashboard")

st.markdown(
    """
Welcome to **RetailPulse**, an end-to-end **Retail Analytics Dashboard**
built using **Python, Streamlit, Machine Learning, and Data Visualization**.

This project demonstrates a complete data science workflow—from data preprocessing
and exploratory analysis to predictive modeling and interactive business dashboards.
"""
)

st.divider()

# ----------------------------------------
# Project Overview
# ----------------------------------------
st.header("📌 Project Overview")

st.write(
    """
RetailPulse enables retailers to gain actionable insights from transactional data.

The dashboard includes:

- Exploratory Data Analysis (EDA)
- Sales Forecasting
- Customer Segmentation using RFM & K-Means
- Customer Churn Prediction
- Inventory & Model Performance Analysis

Each module is designed to provide business intelligence through interactive
visualizations and machine learning models.
"""
)

# ----------------------------------------
# Dashboard Modules
# ----------------------------------------
st.header("📊 Dashboard Modules")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 📊 Exploratory Data Analysis")

    st.write("""
- Dataset overview
- Missing value analysis
- Revenue analysis
- Customer insights
- Sales trends
- Visual analytics
""")

    st.markdown("### 📈 Sales Forecasting")

    st.write("""
- Monthly sales trend
- Revenue forecasting
- Time-series visualization
- Business forecasting insights
""")

    st.markdown("### 👥 Customer Segmentation")

    st.write("""
- Feature engineering
- RFM analysis
- K-Means clustering
- Customer segment profiling
""")

with col2:

    st.markdown("### 🚪 Customer Churn")

    st.write("""
- Customer churn dataset
- Feature selection
- Classification model
- Prediction metrics
""")

    st.markdown("### 📦 Inventory Analytics")

    st.write("""
- Model evaluation
- Confusion Matrix
- ROC Curve
- Feature Importance
- Performance metrics
""")

    st.markdown("### 📁 Generated Assets")

    st.write("""
- Clean dataset
- Feature-engineered dataset
- RFM dataset
- Customer segments
- Churn dataset
- Trained ML models
""")

st.divider()

# ----------------------------------------
# Technology Stack
# ----------------------------------------
st.header("🛠 Technology Stack")

tech1, tech2, tech3 = st.columns(3)

with tech1:

    st.subheader("Programming")

    st.write("""
- Python
- Pandas
- NumPy
""")

with tech2:

    st.subheader("Visualization")

    st.write("""
- Matplotlib
- Seaborn
- Streamlit
""")

with tech3:

    st.subheader("Machine Learning")

    st.write("""
- Scikit-learn
- Joblib
- OpenPyXL
""")

st.divider()

# ----------------------------------------
# Project Workflow
# ----------------------------------------
st.header("⚙ Project Workflow")

st.markdown(
"""
1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Exploratory Data Analysis
5. Sales Forecasting
6. Customer Segmentation (RFM + K-Means)
7. Customer Churn Prediction
8. Inventory & Model Evaluation
9. Dashboard Visualization
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

st.divider()

# ----------------------------------------
# Dashboard Statistics
# ----------------------------------------
st.header("📈 Dashboard Summary")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Modules", "5")
c2.metric("ML Models", "3")
c3.metric("Datasets", "4")
c4.metric("Visualizations", "20+")
c5.metric("Framework", "Streamlit")

st.divider()

# ----------------------------------------
# Key Features
# ----------------------------------------
st.header("✨ Key Highlights")

left, right = st.columns(2)

with left:
    st.success("Interactive Business Dashboard")
    st.success("Customer Segmentation using K-Means")
    st.success("Customer Churn Prediction")
    st.success("Sales Forecasting")

with right:
    st.success("Comprehensive EDA")
    st.success("Inventory Analytics")
    st.success("Machine Learning Pipeline")
    st.success("Professional Data Visualizations")

st.divider()

# ----------------------------------------
# Footer
# ----------------------------------------
st.info(
    """
**RetailPulse** is an end-to-end Retail Analytics project demonstrating
data preprocessing, exploratory data analysis, customer segmentation,
sales forecasting, churn prediction, and inventory analysis using
Python, Machine Learning, and Streamlit.
"""
)

st.caption("© 2026 RetailPulse | Developed using Python • Streamlit • Scikit-learn")
