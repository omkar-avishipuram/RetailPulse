import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Inventory Analysis",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Inventory Analysis")
st.markdown("Inventory & Customer Churn Model Evaluation Dashboard")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
DATA_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
    / "customer_churn.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.success("Customer churn dataset loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Dataset Summary
# --------------------------------------------------
st.header("Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("Features", len(df.columns))
col3.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")

# --------------------------------------------------
# Prepare Features
# --------------------------------------------------
st.header("Prepare Training Data")

X = df[["DaysSinceLastPurchase"]]
y = df["Churn"]

st.success("Features prepared successfully!")

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

col1, col2 = st.columns(2)

col1.metric("Training Samples", len(X_train))
col2.metric("Testing Samples", len(X_test))

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
y_prob = model.predict_proba(X_test)[:, 1]

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.dataframe(prediction_df.head(20))

# --------------------------------------------------
# Model Metrics
# --------------------------------------------------
st.header("Model Performance")

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

col1, col2 = st.columns(2)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "ROC-AUC",
    f"{roc_auc:.3f}"
)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.header("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = (
    pd.DataFrame(report)
    .transpose()
    .round(3)
)

st.dataframe(report_df)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.header("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Active", "Churn"],
    yticklabels=["Active", "Churn"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# --------------------------------------------------
# ROC Curve
# --------------------------------------------------
st.header("ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(6,5))

ax.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}",
    linewidth=2
)

ax.plot([0,1],[0,1],"r--")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

st.dataframe(importance)

fig, ax = plt.subplots(figsize=(8,4))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    palette="viridis",
    ax=ax
)

ax.set_title("Feature Importance")

st.pyplot(fig)

# --------------------------------------------------
# Save Model
# --------------------------------------------------
st.header("Save Trained Model")

models_dir = (
    Path(__file__).parents[2]
    / "models"
)

models_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_path = models_dir / "churn_model.pkl"

joblib.dump(
    model,
    model_path
)

st.success(f"Model saved successfully!\n\n{model_path}")

# --------------------------------------------------
# Save Evaluation Metrics
# --------------------------------------------------
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        roc_auc
    ]
})

st.subheader("Evaluation Metrics")

st.dataframe(metrics)

metrics_path = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
    / "inventory_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

st.success("Evaluation metrics saved successfully!")

# --------------------------------------------------
# Download Metrics
# --------------------------------------------------
with open(metrics_path, "rb") as file:
    st.download_button(
        label="📥 Download Metrics",
        data=file,
        file_name="inventory_metrics.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Dashboard Summary
# --------------------------------------------------
st.header("Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Customers",
    len(df)
)

col2.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col3.metric(
    "ROC-AUC",
    f"{roc_auc:.3f}"
)

st.success("✅ Inventory Analysis completed successfully!")