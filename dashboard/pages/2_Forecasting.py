import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Demand Forecasting")
st.markdown("Monthly Sales Forecast using Holt-Winters Exponential Smoothing")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
DATA_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
    / "online_retail_clean.csv"
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

# ----------------------------------------
# Monthly Sales
# ----------------------------------------
st.header("Monthly Sales")

df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

monthly_sales = (
    df.groupby("Month")["TotalAmount"]
    .sum()
    .reset_index()
)

st.dataframe(monthly_sales.tail())

# ----------------------------------------
# Train Forecasting Model
# ----------------------------------------
st.header("Training Forecast Model")

series = monthly_sales.set_index("Month")["TotalAmount"]

model = ExponentialSmoothing(
    series,
    trend="add",
    seasonal=None
)

fit = model.fit()

st.success("Forecast model trained successfully!")

# ----------------------------------------
# Forecast Next 6 Months
# ----------------------------------------
forecast = fit.forecast(6)

forecast_df = pd.DataFrame({
    "Month": forecast.index,
    "Forecast": forecast.values
})

st.subheader("Next 6 Months Forecast")

st.dataframe(forecast_df)

# ----------------------------------------
# Model Accuracy
# ----------------------------------------
predicted = fit.fittedvalues

mae = mean_absolute_error(series, predicted)
rmse = mean_squared_error(series, predicted) ** 0.5

st.header("Model Performance")

col1, col2 = st.columns(2)

col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")

# ----------------------------------------
# Sales Trend
# ----------------------------------------
st.header("Historical Monthly Sales")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    monthly_sales["Month"],
    monthly_sales["TotalAmount"],
    marker="o",
    label="Actual"
)

ax.set_title("Monthly Sales")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.tick_params(axis="x", rotation=45)
ax.legend()

st.pyplot(fig)

# ----------------------------------------
# Forecast Plot
# ----------------------------------------
st.header("Forecast")

fig2, ax2 = plt.subplots(figsize=(12,5))

ax2.plot(
    series.index,
    series.values,
    label="Historical",
    marker="o"
)

ax2.plot(
    forecast.index,
    forecast.values,
    label="Forecast",
    marker="o",
    linestyle="--",
    color="red"
)

ax2.set_title("Demand Forecast")
ax2.set_xlabel("Month")
ax2.set_ylabel("Revenue")
ax2.tick_params(axis="x", rotation=45)
ax2.legend()

st.pyplot(fig2)

# ----------------------------------------
# Save Forecast
# ----------------------------------------
OUTPUT_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "processed"
    / "sales_forecast.csv"
)

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)

st.success("Forecast saved successfully!")

# ----------------------------------------
# Summary
# ----------------------------------------
st.header("Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Transactions", len(df))
col2.metric("Months", len(monthly_sales))
col3.metric(
    "Total Revenue",
    f"${df['TotalAmount'].sum():,.2f}"
)

st.success("Demand Forecasting completed successfully!")
