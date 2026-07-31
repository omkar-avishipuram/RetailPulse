import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from io import StringIO

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")
st.markdown("RetailPulse - Online Retail Dataset")

# ----------------------------
# Load Data
# ----------------------------
DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "Online Retail.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    return df

df = load_data()

# ----------------------------
# Dataset Overview
# ----------------------------
st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("**Shape**")
    st.write(df.shape)

    st.write("**Columns**")
    st.write(df.columns.tolist())

with col2:
    st.write("**Missing Values**")
    st.dataframe(df.isnull().sum().to_frame("Missing"))

    st.write("**Duplicate Rows**")
    st.write(df.duplicated().sum())

st.subheader("Preview")
st.dataframe(df.head())

st.subheader("Statistical Summary")
st.dataframe(df.describe())

st.subheader("Dataset Information")
buffer = StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

# ----------------------------
# Data Cleaning
# ----------------------------
st.header("Data Cleaning")

df_clean = df.copy()

df_clean = df_clean.dropna(subset=["CustomerID"])
df_clean = df_clean.drop_duplicates()

df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

df_clean = df_clean[
    ~df_clean["InvoiceNo"].astype(str).str.startswith("C")
]

df_clean = df_clean[
    (df_clean["Quantity"] > 0)
    & (df_clean["UnitPrice"] > 0)
]

df_clean["TotalAmount"] = (
    df_clean["Quantity"] * df_clean["UnitPrice"]
)

st.success("Dataset cleaned successfully!")

st.write("Clean Dataset Shape:", df_clean.shape)

# Save cleaned dataset
processed_path = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
    / "online_retail_clean.csv"
)

processed_path.parent.mkdir(parents=True, exist_ok=True)

df_clean.to_csv(processed_path, index=False)

# ----------------------------
# Visualizations
# ----------------------------
sns.set_style("whitegrid")

st.header("Visualizations")

# Monthly Sales
df_clean["Month"] = (
    df_clean["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    df_clean.groupby("Month")["TotalAmount"]
    .sum()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14,6))
sns.lineplot(
    data=monthly_sales,
    x="Month",
    y="TotalAmount",
    marker="o",
    ax=ax
)
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
st.pyplot(fig)

# Top Products
top_products = (
    df_clean.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(
    x=top_products.values,
    y=top_products.index,
    ax=ax
)
plt.title("Top 10 Selling Products")
plt.xlabel("Quantity Sold")
st.pyplot(fig)

# Top Countries
top_countries = (
    df_clean.groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    ax=ax
)
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
st.pyplot(fig)

# Sales Distribution
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(
    df_clean["TotalAmount"],
    bins=50,
    kde=True,
    ax=ax
)
plt.title("Distribution of Sales Amount")
st.pyplot(fig)

# Quantity Distribution
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(
    x=df_clean["Quantity"],
    ax=ax
)
plt.title("Order Quantity Distribution")
st.pyplot(fig)

# Top Customers
top_customers = (
    df_clean.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(
    x=top_customers.index.astype(str),
    y=top_customers.values,
    ax=ax
)
plt.xticks(rotation=45)
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer ID")
plt.ylabel("Revenue")
st.pyplot(fig)

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(
    df_clean[
        ["Quantity", "UnitPrice", "TotalAmount"]
    ].corr(),
    annot=True,
    cmap="Blues",
    ax=ax
)
plt.title("Correlation Heatmap")
st.pyplot(fig)

st.success("EDA completed successfully!")