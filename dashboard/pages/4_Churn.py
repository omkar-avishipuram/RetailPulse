import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn Prediction")
st.markdown(
    "Predict customer churn using Machine Learning based on customer purchase behaviour."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
BASE_DIR = Path(__file__).parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "online_retail_features.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


df = load_data()

st.success("✅ Dataset loaded successfully!")

# --------------------------------------------------
# Dataset Summary
# --------------------------------------------------
st.header("Dataset Summary")

c1, c2, c3 = st.columns(3)

c1.metric("Transactions", f"{len(df):,}")
c2.metric("Customers", f"{df['CustomerID'].nunique():,}")
c3.metric("Revenue", f"${df['TotalAmount'].sum():,.2f}")

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# Create Churn Labels
# --------------------------------------------------
st.header("Customer Churn Label Generation")

last_purchase = (
    df.groupby("CustomerID")["InvoiceDate"]
    .max()
    .reset_index()
)

reference_date = df["InvoiceDate"].max()

last_purchase["DaysSinceLastPurchase"] = (
    reference_date - last_purchase["InvoiceDate"]
).dt.days

last_purchase["Churn"] = (
    last_purchase["DaysSinceLastPurchase"] > 90
).astype(int)

st.success("✅ Churn labels generated successfully.")

st.dataframe(last_purchase.head())

# --------------------------------------------------
# Customer-Level Feature Engineering
# --------------------------------------------------
st.header("Customer Feature Engineering")

customer_df = (
    df.groupby("CustomerID")
    .agg(
        TotalAmount=("TotalAmount", "sum"),
        Quantity=("Quantity", "sum"),
        NumPurchases=("InvoiceNo", "nunique"),
    )
    .reset_index()
)

customer_df = customer_df.merge(
    last_purchase[
        [
            "CustomerID",
            "Churn",
        ]
    ],
    on="CustomerID",
)

st.success("✅ Customer-level dataset created.")

st.dataframe(customer_df.head(), use_container_width=True)
# --------------------------------------------------
# Prepare Training Data
# --------------------------------------------------
st.header("Prepare Training Data")

X = customer_df[
    [
        "TotalAmount",
        "Quantity",
        "NumPurchases",
    ]
]

y = customer_df["Churn"]

c1, c2 = st.columns(2)

c1.metric("Features", X.shape[1])
c2.metric("Customers", len(customer_df))

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

st.success("✅ Training and testing datasets created.")

c1, c2 = st.columns(2)

c1.metric("Training Samples", len(X_train))
c2.metric("Testing Samples", len(X_test))

# --------------------------------------------------
# Train Random Forest Model
# --------------------------------------------------
st.header("Random Forest Model")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)

st.success("✅ Model trained successfully!")

# --------------------------------------------------
# Predictions
# --------------------------------------------------
st.header("Model Predictions")

y_pred = model.predict(X_test)

prediction_df = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": y_pred,
    }
)

st.dataframe(
    prediction_df.head(20),
    use_container_width=True,
)

# --------------------------------------------------
# Performance Metrics
# --------------------------------------------------
st.header("Model Performance")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{accuracy:.2%}")
m2.metric("Precision", f"{precision:.2%}")
m3.metric("Recall", f"{recall:.2%}")
m4.metric("F1 Score", f"{f1:.2%}")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.header("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Active", "Churn"],
    yticklabels=["Active", "Churn"],
    ax=ax,
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.header("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)

report_df = (
    pd.DataFrame(report)
    .transpose()
    .round(3)
)

st.dataframe(
    report_df,
    use_container_width=True,
)
# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.header("Feature Importance")

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }
).sort_values("Importance", ascending=False)

st.dataframe(
    importance,
    use_container_width=True,
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False,
    ax=ax,
)

ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")

st.pyplot(fig)

# --------------------------------------------------
# Customer Churn Distribution
# --------------------------------------------------
st.header("Customer Churn Distribution")

fig, ax = plt.subplots(figsize=(6, 4))

sns.countplot(
    data=customer_df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False,
    ax=ax,
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Active", "Churned"])
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Customer Status")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)

# --------------------------------------------------
# Create Final Customer Churn Dataset
# --------------------------------------------------
st.header("Generated Customer Churn Dataset")

customer_churn = customer_df.copy()

customer_churn["Prediction"] = model.predict(
    customer_churn[
        [
            "TotalAmount",
            "Quantity",
            "NumPurchases",
        ]
    ]
)

# Keep only the columns required by Inventory.py
customer_churn = customer_churn[
    [
        "CustomerID",
        "TotalAmount",
        "Quantity",
        "NumPurchases",
        "Churn",
        "Prediction",
    ]
]

st.dataframe(
    customer_churn.head(20),
    use_container_width=True,
)

# --------------------------------------------------
# Dataset Statistics
# --------------------------------------------------
st.header("Dataset Statistics")

c1, c2, c3 = st.columns(3)

c1.metric("Total Customers", f"{len(customer_churn):,}")
c2.metric(
    "Predicted Churn",
    int(customer_churn["Prediction"].sum()),
)
c3.metric(
    "Predicted Active",
    int((customer_churn["Prediction"] == 0).sum()),
)

# --------------------------------------------------
# Prediction Distribution
# --------------------------------------------------
st.header("Prediction Distribution")

prediction_counts = (
    customer_churn["Prediction"]
    .value_counts()
    .rename(
        index={
            0: "Active",
            1: "Churn",
        }
    )
)

fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(
    prediction_counts.values,
    labels=prediction_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#66BB6A", "#EF5350"],
)

ax.set_title("Predicted Customer Status")

st.pyplot(fig)
# --------------------------------------------------
# Save Customer Churn Dataset
# --------------------------------------------------
st.header("Save Results")

processed_dir = (
    BASE_DIR
    / "data"
    / "processed"
)

processed_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = processed_dir / "customer_churn.csv"

customer_churn.to_csv(
    output_path,
    index=False,
)

# Do NOT display the local filesystem path
st.success("✅ Customer churn dataset saved successfully!")

# --------------------------------------------------
# Download Dataset
# --------------------------------------------------
with open(output_path, "rb") as file:
    st.download_button(
        label="📥 Download Customer Churn Dataset",
        data=file,
        file_name="customer_churn.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# Preview Final Dataset
# --------------------------------------------------
st.header("Final Customer Churn Dataset")

st.dataframe(
    customer_churn,
    use_container_width=True,
)

# --------------------------------------------------
# Workflow Summary
# --------------------------------------------------
st.divider()

st.success("✅ Customer Churn Prediction completed successfully!")

st.info(
    """
### Workflow Completed

- ✅ Loaded feature-engineered dataset
- ✅ Generated customer churn labels
- ✅ Created customer-level features
- ✅ Trained Random Forest classifier
- ✅ Evaluated model performance
- ✅ Displayed confusion matrix
- ✅ Generated classification report
- ✅ Visualized feature importance
- ✅ Saved `customer_churn.csv`
- ✅ Enabled dataset download
"""
)