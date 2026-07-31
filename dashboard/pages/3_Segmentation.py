import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation Dashboard")
st.markdown("### Feature Engineering + RFM Analysis + K-Means Clustering")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).parents[2]

RAW_DATA = BASE_DIR / "data" / "processed" / "online_retail_clean.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

RFM_PATH = PROCESSED_DIR / "rfm.csv"
FEATURE_PATH = PROCESSED_DIR / "online_retail_features.csv"
SEGMENT_PATH = PROCESSED_DIR / "customer_segments.csv"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(RAW_DATA)

df = load_data()

st.success("Dataset loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------
st.header("Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("Missing Values")
st.dataframe(df.isnull().sum().to_frame("Missing"))

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
st.header("Feature Engineering")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Quarter"] = df["InvoiceDate"].dt.quarter
df["Day"] = df["InvoiceDate"].dt.day
df["Weekday"] = df["InvoiceDate"].dt.day_name()
df["Hour"] = df["InvoiceDate"].dt.hour
df["Weekend"] = df["InvoiceDate"].dt.dayofweek >= 5

st.success("Date features created successfully.")

st.subheader("Feature Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Create RFM Dataset
# --------------------------------------------------
st.header("RFM Analysis")

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalAmount": "sum"
})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

st.subheader("RFM Dataset")

st.dataframe(rfm.head())

rfm.to_csv(RFM_PATH)

df.to_csv(
    FEATURE_PATH,
    index=False
)

st.success("RFM dataset saved successfully.")

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------
st.header("Feature Scaling")

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm[["Recency", "Frequency", "Monetary"]]
)

st.success("Features scaled successfully.")

# --------------------------------------------------
# Elbow Method
# --------------------------------------------------
st.header("Elbow Method")

wcss = []

for i in range(1, 11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)

    wcss.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    range(1,11),
    wcss,
    marker="o"
)

ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")

st.pyplot(fig)

# --------------------------------------------------
# Silhouette Scores
# --------------------------------------------------
st.header("Silhouette Scores")

scores = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(rfm_scaled)

    score = silhouette_score(
        rfm_scaled,
        labels
    )

    scores.append({
        "Clusters": k,
        "Silhouette Score": round(score,3)
    })

st.dataframe(pd.DataFrame(scores))

# --------------------------------------------------
# Train Final Model
# --------------------------------------------------
st.header("K-Means Clustering")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

st.success("Customer segmentation completed successfully.")

# --------------------------------------------------
# Cluster Summary
# --------------------------------------------------
st.header("Cluster Summary")

cluster_summary = (
    rfm.groupby("Cluster")
    [["Recency","Frequency","Monetary"]]
    .mean()
)

st.dataframe(cluster_summary)

# --------------------------------------------------
# Customer Segments
# --------------------------------------------------
st.header("Customer Segments")

fig, ax = plt.subplots(figsize=(10,6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="Set2",
    s=80,
    ax=ax
)

ax.set_title("Customer Segments")
ax.set_xlabel("Frequency")
ax.set_ylabel("Monetary")

st.pyplot(fig)

# --------------------------------------------------
# Scaled Cluster Visualization
# --------------------------------------------------
st.header("Scaled Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(10,7))

scatter = ax2.scatter(
    rfm_scaled[:,0],
    rfm_scaled[:,1],
    c=rfm["Cluster"],
    cmap="viridis",
    s=40
)

ax2.set_title("Customer Segmentation using K-Means")
ax2.set_xlabel("Scaled Recency")
ax2.set_ylabel("Scaled Frequency")

plt.colorbar(
    scatter,
    ax=ax2,
    label="Cluster"
)

fig2.savefig(
    SCREENSHOT_DIR / "customer_segmentation.png",
    dpi=300,
    bbox_inches="tight"
)

st.pyplot(fig2)

# --------------------------------------------------
# Save Segments
# --------------------------------------------------
rfm.to_csv(
    SEGMENT_PATH,
    index=True
)

st.success("Customer segments saved successfully.")

# --------------------------------------------------
# Final Dataset Preview
# --------------------------------------------------
st.header("Segmented Customers")

st.dataframe(rfm.head(15))

# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------
st.header("Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    len(rfm)
)

col2.metric(
    "Clusters",
    rfm["Cluster"].nunique()
)

col3.metric(
    "Average Revenue",
    f"${rfm['Monetary'].mean():,.2f}"
)

col4.metric(
    "Highest Frequency",
    int(rfm["Frequency"].max())
)

st.success("🎉 Customer Segmentation Dashboard completed successfully!")