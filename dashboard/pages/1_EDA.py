import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from io import StringIO

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Exploratory Data Analysis")
st.markdown("### RetailPulse - Online Retail Dataset Analysis")

# ----------------------------
# Load Data
# ----------------------------
DATA_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "raw"
    / "Online Retail.xlsx"
)


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    return df


df = load_data()

# ----------------------------
# Dataset Summary
# ----------------------------
st.header("📋 Dataset Summary")

rows, cols = df.shape
missing = int(df.isnull().sum().sum())
duplicates = int(df.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{rows:,}")
c2.metric("Columns", cols)
c3.metric("Missing Values", f"{missing:,}")
c4.metric("Duplicate Rows", f"{duplicates:,}")

st.divider()

# ----------------------------
# Missing Values
# ----------------------------
st.subheader("Missing Values")

missing_df = (
    df.isnull()
    .sum()
    .reset_index()
)

missing_df.columns = ["Column", "Missing Values"]

missing_df = missing_df.sort_values(
    by="Missing Values",
    ascending=False
)

st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True,
)

# ----------------------------
# Dataset Preview
# ----------------------------
st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
)

# ----------------------------
# Statistical Summary
# ----------------------------
st.subheader("Statistical Summary")

st.dataframe(
    df.describe(include="all"),
    use_container_width=True,
)

# ----------------------------
# Dataset Information
# ----------------------------
st.subheader("Dataset Information")

buffer = StringIO()
df.info(buf=buffer)

st.code(
    buffer.getvalue(),
    language="text",
)

# ----------------------------
# Data Cleaning
# ----------------------------
st.header("🧹 Data Cleaning")

df_clean = df.copy()

df_clean = df_clean.dropna(subset=["CustomerID"])
df_clean = df_clean.drop_duplicates()

df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

# Remove cancelled invoices
df_clean = df_clean[
    ~df_clean["InvoiceNo"].astype(str).str.startswith("C")
]

# Remove invalid values
df_clean = df_clean[
    (df_clean["Quantity"] > 0)
    & (df_clean["UnitPrice"] > 0)
]

# Feature Engineering
df_clean["TotalAmount"] = (
    df_clean["Quantity"] * df_clean["UnitPrice"]
)

st.success("✅ Dataset cleaned successfully!")

clean_rows, clean_cols = df_clean.shape

cc1, cc2 = st.columns(2)

cc1.metric("Clean Rows", f"{clean_rows:,}")
cc2.metric("Clean Columns", clean_cols)

# ----------------------------
# Save Clean Dataset
# ----------------------------
processed_path = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
    / "online_retail_clean.csv"
)

processed_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df_clean.to_csv(
    processed_path,
    index=False,
)

st.info("Clean dataset saved to data/processed/online_retail_clean.csv")

# ----------------------------
# Visualizations
# ----------------------------
st.header("📈 Data Visualizations")

sns.set_style("whitegrid")

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

fig, ax = plt.subplots(figsize=(14, 6))

sns.lineplot(
    data=monthly_sales,
    x="Month",
    y="TotalAmount",
    marker="o",
    linewidth=2,
    ax=ax,
)

ax.set_title("Monthly Sales Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
plt.xticks(rotation=45)

st.pyplot(fig)

# ----------------------------
# Top Products
# ----------------------------
top_products = (
    df_clean.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    x=top_products.values,
    y=top_products.index,
    palette="viridis",
    ax=ax,
)

ax.set_title("Top 10 Selling Products")
ax.set_xlabel("Quantity Sold")
ax.set_ylabel("Product")

st.pyplot(fig)

# ----------------------------
# Top Countries
# ----------------------------
top_countries = (
    df_clean.groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    palette="magma",
    ax=ax,
)

ax.set_title("Top 10 Countries by Revenue")
ax.set_xlabel("Revenue")

st.pyplot(fig)

# ----------------------------
# Sales Distribution
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    df_clean["TotalAmount"],
    bins=50,
    kde=True,
    color="steelblue",
    ax=ax,
)

ax.set_title("Sales Amount Distribution")

st.pyplot(fig)

# ----------------------------
# Quantity Distribution
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(
    x=df_clean["Quantity"],
    color="orange",
    ax=ax,
)

ax.set_title("Order Quantity Distribution")

st.pyplot(fig)

# ----------------------------
# Top Customers
# ----------------------------
top_customers = (
    df_clean.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    x=top_customers.index.astype(str),
    y=top_customers.values,
    palette="crest",
    ax=ax,
)

ax.set_title("Top 10 Customers by Revenue")
ax.set_xlabel("Customer ID")
ax.set_ylabel("Revenue")

plt.xticks(rotation=45)

st.pyplot(fig)

# ----------------------------
# Correlation Heatmap
# ----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

corr = df_clean[
    [
        "Quantity",
        "UnitPrice",
        "TotalAmount",
    ]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    fmt=".2f",
    ax=ax,
)

ax.set_title("Correlation Heatmap")

st.pyplot(fig)

# ----------------------------
# Dashboard Summary
# ----------------------------
st.header("📌 Dashboard Summary")

s1, s2, s3, s4 = st.columns(4)

s1.metric("Transactions", f"{len(df_clean):,}")
s2.metric("Customers", f"{df_clean['CustomerID'].nunique():,}")
s3.metric("Products", f"{df_clean['StockCode'].nunique():,}")
s4.metric(
    "Revenue",
    f"${df_clean['TotalAmount'].sum():,.2f}",
)

st.divider()

st.success("✅ Exploratory Data Analysis completed successfully!")