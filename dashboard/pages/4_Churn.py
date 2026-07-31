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
    layout="wide"
)

st.title("📉 Customer Churn Prediction")
st.markdown(
    "Predict customer churn using machine learning based on purchase behavior."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
DATA_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
    / "online_retail_features.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

df = load_data()

st.success("Dataset loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Dataset Summary
# --------------------------------------------------
st.header("Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Transactions", len(df))
col2.metric("Customers", df["CustomerID"].nunique())
col3.metric("Revenue", f"${df['TotalAmount'].sum():,.2f}")

# --------------------------------------------------
# Create Churn Labels
# --------------------------------------------------
st.header("Create Customer Churn Labels")

last_purchase = (
    df.groupby("CustomerID")["InvoiceDate"]
      .max()
      .reset_index()
)

last_purchase.columns = [
    "CustomerID",
    "LastPurchase"
]

reference_date = df["InvoiceDate"].max()

last_purchase["DaysSinceLastPurchase"] = (
    reference_date - last_purchase["LastPurchase"]
).dt.days

last_purchase["Churn"] = (
    last_purchase["DaysSinceLastPurchase"] > 90
).astype(int)

st.success("Churn labels generated successfully!")

st.subheader("Customer Churn Preview")
st.dataframe(last_purchase.head())
# --------------------------------------------------
# Customer-Level Feature Engineering
# --------------------------------------------------
st.header("Customer-Level Feature Engineering")

customer_df = (
    df.groupby("CustomerID")
    .agg({
        "TotalAmount": "sum",
        "Quantity": "sum",
        "InvoiceNo": "nunique"
    })
    .reset_index()
)

customer_df.rename(
    columns={"InvoiceNo": "NumPurchases"},
    inplace=True
)

customer_df = customer_df.merge(
    last_purchase[
        ["CustomerID", "Churn"]
    ],
    on="CustomerID"
)

st.success("Customer-level dataset created!")

st.subheader("Customer Dataset Preview")
st.dataframe(customer_df.head())

# --------------------------------------------------
# Feature Matrix
# --------------------------------------------------
st.header("Prepare Training Data")

X = customer_df[
    [
        "TotalAmount",
        "Quantity",
        "NumPurchases"
    ]
]

y = customer_df["Churn"]

col1, col2 = st.columns(2)

with col1:
    st.metric("Features", X.shape[1])

with col2:
    st.metric("Customers", len(customer_df))

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

st.success("Dataset split into training and testing sets.")

col1, col2 = st.columns(2)

with col1:
    st.metric("Training Samples", len(X_train))

with col2:
    st.metric("Testing Samples", len(X_test))

# --------------------------------------------------
# Train Random Forest Model
# --------------------------------------------------
st.header("Train Random Forest Model")

model = RandomForestClassifier(
    random_state=42,
    n_estimators=100
)

model.fit(X_train, y_train)

st.success("Model trained successfully!")

# --------------------------------------------------
# Predictions
# --------------------------------------------------
st.header("Predictions")

y_pred = model.predict(X_test)

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.dataframe(prediction_df.head(20))

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
st.header("Model Performance")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "Precision",
    f"{precision:.2%}"
)

col3.metric(
    "Recall",
    f"{recall:.2%}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2%}"
)
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
    xticklabels=["Not Churn", "Churn"],
    yticklabels=["Not Churn", "Churn"],
    ax=ax
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
    zero_division=0
)

report_df = (
    pd.DataFrame(report)
    .transpose()
    .round(3)
)

st.dataframe(report_df)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance)

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    palette="viridis",
    ax=ax
)

ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")

st.pyplot(fig)

# --------------------------------------------------
# Churn Distribution
# --------------------------------------------------
st.header("Customer Churn Distribution")

fig, ax = plt.subplots(figsize=(6, 4))

sns.countplot(
    data=customer_df,
    x="Churn",
    palette="Set2",
    ax=ax
)

ax.set_xticklabels(["Active", "Churned"])
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Customer Status")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)

# --------------------------------------------------
# Customer Churn Dataset
# --------------------------------------------------
st.header("Customer Churn Dataset")

customer_churn = customer_df.copy()

customer_churn["Prediction"] = model.predict(
    customer_df[
        [
            "TotalAmount",
            "Quantity",
            "NumPurchases"
        ]
    ]
)

st.dataframe(customer_churn.head(20))
# --------------------------------------------------
# Save Customer Churn Dataset
# --------------------------------------------------
st.header("Save Results")

processed_dir = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
)

processed_dir.mkdir(parents=True, exist_ok=True)

output_path = processed_dir / "customer_churn.csv"

customer_churn.to_csv(
    output_path,
    index=False
)

st.success(f"Customer churn dataset saved successfully!\n\n{output_path}")

# --------------------------------------------------
# Download Button
# --------------------------------------------------
with open(output_path, "rb") as file:
    st.download_button(
        label="📥 Download Customer Churn Dataset",
        data=file,
        file_name="customer_churn.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Dataset Statistics
# --------------------------------------------------
st.header("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    len(customer_churn)
)

col2.metric(
    "Predicted Churn",
    int(customer_churn["Prediction"].sum())
)

col3.metric(
    "Predicted Active",
    int((customer_churn["Prediction"] == 0).sum())
)

# --------------------------------------------------
# Churn Percentage
# --------------------------------------------------
st.header("Prediction Distribution")

prediction_counts = (
    customer_churn["Prediction"]
    .value_counts()
    .rename(index={0: "Active", 1: "Churn"})
)

fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(
    prediction_counts.values,
    labels=prediction_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#66BB6A", "#EF5350"]
)

ax.set_title("Predicted Customer Status")

st.pyplot(fig)

# --------------------------------------------------
# Preview Final Dataset
# --------------------------------------------------
st.header("Final Customer Churn Dataset")

st.dataframe(customer_churn)

# --------------------------------------------------
# Footer Summary
# --------------------------------------------------
st.divider()

st.success("✅ Customer Churn Prediction completed successfully!")

st.info(
    """
    **Workflow Completed**
    
    ✔ Loaded feature-engineered dataset
    
    ✔ Created churn labels
    
    ✔ Built customer-level dataset
    
    ✔ Trained Random Forest classifier
    
    ✔ Evaluated model performance
    
    ✔ Visualized confusion matrix
    
    ✔ Generated classification report
    
    ✔ Displayed feature importance
    
    ✔ Saved customer_churn.csv
    
    ✔ Enabled dataset download
    """
)