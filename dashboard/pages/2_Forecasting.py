import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Demand Forecasting")
st.markdown(
    """
Forecast future retail sales using the pre-generated Prophet forecast.
This dashboard visualizes historical sales trends and forecasted demand.
    """
)

# --------------------------------------------------
# File Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DAILY_PATH = PROCESSED_DIR / "daily_sales.csv"
FORECAST_PATH = PROCESSED_DIR / "prophet_forecast.csv"

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    daily_df = pd.read_csv(DAILY_PATH)
    forecast_df = pd.read_csv(FORECAST_PATH)

    daily_df["InvoiceDate"] = pd.to_datetime(daily_df["InvoiceDate"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    return daily_df, forecast_df

daily_df, forecast_df = load_data()

st.success("Datasets loaded successfully!")

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.header("Dataset Preview")

tab1, tab2 = st.tabs(["Daily Sales", "Forecast"])

with tab1:
    st.dataframe(daily_df.head())

with tab2:
    st.dataframe(forecast_df.head())

# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------
st.header("Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Historical Records", len(daily_df))
c2.metric("Forecast Records", len(forecast_df))
c3.metric(
    "Total Revenue",
    f"${daily_df['TotalAmount'].sum():,.0f}"
)
c4.metric(
    "Average Daily Sales",
    f"${daily_df['TotalAmount'].mean():,.2f}"
)

# --------------------------------------------------
# Historical Sales
# --------------------------------------------------
st.header("Historical Daily Sales")

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(
    daily_df["InvoiceDate"],
    daily_df["TotalAmount"],
    color="steelblue",
    linewidth=1.5,
    label="Daily Sales"
)

ax.set_title("Daily Sales Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)
# --------------------------------------------------
# Rolling Statistics
# --------------------------------------------------
st.header("📊 7-Day Rolling Statistics")

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(
    daily_df["InvoiceDate"],
    daily_df["TotalAmount"],
    color="lightgray",
    linewidth=1,
    alpha=0.6,
    label="Daily Sales"
)

ax.plot(
    daily_df["InvoiceDate"],
    daily_df["RollingMean7"],
    color="blue",
    linewidth=2,
    label="7-Day Rolling Mean"
)

ax.fill_between(
    daily_df["InvoiceDate"],
    daily_df["RollingMean7"] - daily_df["RollingStd7"].fillna(0),
    daily_df["RollingMean7"] + daily_df["RollingStd7"].fillna(0),
    color="skyblue",
    alpha=0.25,
    label="Rolling Std Dev"
)

ax.set_title("Daily Sales with Rolling Mean")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Forecast Dataset Summary
# --------------------------------------------------
st.header("📈 Forecast Summary")

fc1, fc2, fc3 = st.columns(3)

fc1.metric(
    "Forecast Start",
    forecast_df["ds"].min().strftime("%Y-%m-%d")
)

fc2.metric(
    "Forecast End",
    forecast_df["ds"].max().strftime("%Y-%m-%d")
)

fc3.metric(
    "Average Forecast",
    f"${forecast_df['yhat'].mean():,.2f}"
)

# --------------------------------------------------
# Prophet Forecast
# --------------------------------------------------
st.header("🔮 Prophet Sales Forecast")

fig2, ax2 = plt.subplots(figsize=(14,6))

ax2.plot(
    forecast_df["ds"],
    forecast_df["yhat"],
    color="red",
    linewidth=2,
    label="Forecast"
)

ax2.fill_between(
    forecast_df["ds"],
    forecast_df["yhat_lower"],
    forecast_df["yhat_upper"],
    color="salmon",
    alpha=0.30,
    label="Confidence Interval"
)

ax2.set_title("Prophet Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Predicted Sales")
ax2.legend()

st.pyplot(fig2)

# --------------------------------------------------
# Historical vs Forecast
# --------------------------------------------------
st.header("📉 Historical vs Forecast")

fig3, ax3 = plt.subplots(figsize=(14,6))

ax3.plot(
    daily_df["InvoiceDate"],
    daily_df["TotalAmount"],
    color="steelblue",
    linewidth=2,
    label="Historical Sales"
)

ax3.plot(
    forecast_df["ds"],
    forecast_df["yhat"],
    color="crimson",
    linestyle="--",
    linewidth=2,
    label="Forecast"
)

ax3.set_title("Historical Sales vs Forecast")
ax3.set_xlabel("Date")
ax3.set_ylabel("Sales")
ax3.legend()

st.pyplot(fig3)
# --------------------------------------------------
# Forecast Table
# --------------------------------------------------
st.header("📋 Forecast Data")

forecast_display = forecast_df[[
    "ds",
    "yhat",
    "yhat_lower",
    "yhat_upper"
]].copy()

forecast_display.columns = [
    "Date",
    "Forecast",
    "Lower Bound",
    "Upper Bound"
]

st.dataframe(
    forecast_display,
    use_container_width=True
)

# --------------------------------------------------
# Forecast Statistics
# --------------------------------------------------
st.header("📊 Forecast Statistics")

sc1, sc2, sc3, sc4 = st.columns(4)

sc1.metric(
    "Minimum Forecast",
    f"${forecast_df['yhat'].min():,.2f}"
)

sc2.metric(
    "Maximum Forecast",
    f"${forecast_df['yhat'].max():,.2f}"
)

sc3.metric(
    "Average Forecast",
    f"${forecast_df['yhat'].mean():,.2f}"
)

sc4.metric(
    "Forecast Days",
    len(forecast_df)
)

# --------------------------------------------------
# Monthly Forecast Summary
# --------------------------------------------------
st.header("📅 Monthly Forecast Summary")

monthly_forecast = forecast_df.copy()

monthly_forecast["Month"] = (
    monthly_forecast["ds"]
    .dt.to_period("M")
    .astype(str)
)

monthly_summary = (
    monthly_forecast
    .groupby("Month")["yhat"]
    .sum()
    .reset_index()
)

st.dataframe(
    monthly_summary,
    use_container_width=True
)

# --------------------------------------------------
# Monthly Forecast Chart
# --------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(12,5))

ax4.bar(
    monthly_summary["Month"],
    monthly_summary["yhat"],
    color="royalblue"
)

ax4.set_title("Monthly Forecasted Sales")
ax4.set_xlabel("Month")
ax4.set_ylabel("Forecast Sales")
ax4.tick_params(axis="x", rotation=45)

st.pyplot(fig4)

# --------------------------------------------------
# Download Forecast
# --------------------------------------------------
st.header("📥 Download Forecast")

csv = forecast_display.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)
# --------------------------------------------------
# Save Forecast
# --------------------------------------------------
st.header("💾 Save Forecast")

OUTPUT_PATH = PROCESSED_DIR / "sales_forecast.csv"

forecast_display.to_csv(
    OUTPUT_PATH,
    index=False
)

st.success(f"Forecast saved successfully to:\n\n{OUTPUT_PATH}")

# --------------------------------------------------
# Forecast Insights
# --------------------------------------------------
st.header("📌 Forecast Insights")

max_idx = forecast_df["yhat"].idxmax()
min_idx = forecast_df["yhat"].idxmin()

highest_date = forecast_df.loc[max_idx, "ds"]
highest_value = forecast_df.loc[max_idx, "yhat"]

lowest_date = forecast_df.loc[min_idx, "ds"]
lowest_value = forecast_df.loc[min_idx, "yhat"]

col1, col2 = st.columns(2)

with col1:
    st.info(
        f"""
### 📈 Peak Forecast

**Date:** {highest_date.strftime('%d %b %Y')}

**Predicted Sales:** ${highest_value:,.2f}
"""
    )

with col2:
    st.info(
        f"""
### 📉 Lowest Forecast

**Date:** {lowest_date.strftime('%d %b %Y')}

**Predicted Sales:** ${lowest_value:,.2f}
"""
    )

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------
st.header("📂 Dataset Information")

info_df = pd.DataFrame(
    {
        "Dataset": [
            "daily_sales.csv",
            "prophet_forecast.csv",
            "sales_forecast.csv",
        ],
        "Rows": [
            len(daily_df),
            len(forecast_df),
            len(forecast_display),
        ],
        "Columns": [
            daily_df.shape[1],
            forecast_df.shape[1],
            forecast_display.shape[1],
        ],
    }
)

st.dataframe(info_df, use_container_width=True)

# --------------------------------------------------
# Dashboard Summary
# --------------------------------------------------
st.header("📊 Dashboard Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Historical Records",
    f"{len(daily_df):,}"
)

m2.metric(
    "Forecast Records",
    f"{len(forecast_df):,}"
)

m3.metric(
    "Forecast Months",
    monthly_summary.shape[0]
)

m4.metric(
    "Total Historical Revenue",
    f"${daily_df['TotalAmount'].sum():,.0f}"
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.success("✅ Demand Forecasting Dashboard loaded successfully!")

st.caption(
    "RetailPulse • Demand Forecasting using Prophet • Streamlit Dashboard"
)
