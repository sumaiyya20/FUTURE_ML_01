"""
Sales & Demand Forecasting for Businesses
Machine Learning Internship — Task 1

A business-friendly web app that:
  - Cleans historical sales data
  - Engineers time-based features
  - Trains a forecasting model (Random Forest or Linear Regression)
  - Evaluates accuracy / error
  - Visualizes forecasts for non-technical stakeholders
  - Produces a plain-English business summary

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io  (Streamlit Community Cloud)
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    aggregate_daily,
    business_summary,
    clean_data,
    evaluate_model,
    forecast_future,
    make_features,
    train_model,
    train_test_split_by_time,
)

st.set_page_config(
    page_title="Sales & Demand Forecasting",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("📈 Sales & Demand Forecasting for Businesses")
st.caption("Machine Learning Internship — Task 1  |  Predict future sales, explained in plain business terms.")

with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown(
        """
        This tool builds a **sales/demand forecasting model** from historical data and
        presents the results the way a **store owner, startup founder, or business manager**
        would want to see them — not just raw numbers.

        **Pipeline:** data cleaning → time-based feature engineering → regression / time-series
        forecasting → error analysis → business-ready visuals.
        """
    )

# --------------------------------------------------------------------------
# SIDEBAR — DATA INPUT & SETTINGS
# --------------------------------------------------------------------------
st.sidebar.header("1️⃣ Data")

data_source = st.sidebar.radio(
    "Choose a data source",
    ["Use sample dataset", "Upload my own CSV"],
)

if data_source == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV with a date column and a sales/demand column", type="csv")
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
    else:
        st.info("👈 Upload a CSV, or switch to the sample dataset, to get started.")
        st.stop()
else:
    raw_df = pd.read_csv("data/sales_data.csv")
    st.sidebar.success("Loaded sample dataset: 4 years, 3 stores, 3 categories (synthetic).")

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ Column mapping")
cols = list(raw_df.columns)
date_col_guess = next((c for c in cols if "date" in c.lower()), cols[0])
target_col_guess = next((c for c in cols if c.lower() in ("sales", "demand", "quantity", "revenue")), cols[-1])

date_col = st.sidebar.selectbox("Date column", cols, index=cols.index(date_col_guess))
target_col = st.sidebar.selectbox(
    "Sales / demand column", [c for c in cols if c != date_col],
    index=max(0, [c for c in cols if c != date_col].index(target_col_guess)) if target_col_guess in cols and target_col_guess != date_col else 0,
)

st.sidebar.markdown("---")
st.sidebar.header("3️⃣ Model settings")
model_type = st.sidebar.selectbox(
    "Forecasting method", ["random_forest", "linear_regression"],
    format_func=lambda x: "Random Forest (recommended)" if x == "random_forest" else "Linear Regression (simple trend)",
)
horizon = st.sidebar.slider("Days to forecast into the future", 7, 90, 30)
run_btn = st.sidebar.button("🚀 Run Forecast", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built for the Future Interns — ML Task 1 (2026): Sales & Demand Forecasting for Businesses.")

# --------------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------------
if run_btn:
    with st.spinner("Cleaning data, engineering features, and training the model..."):
        cleaned = clean_data(raw_df, date_col, target_col)
        daily = aggregate_daily(cleaned, date_col, target_col)
        daily = daily.rename(columns={date_col: "date", target_col: "sales"}) if date_col != "date" or target_col != "sales" else daily
        # Ensure standardized names regardless of mapping
        daily.columns = ["date", "sales"]

        featured = make_features(daily, "date", "sales").dropna().reset_index(drop=True)
        train_df, test_df = train_test_split_by_time(featured, test_size=0.15)

        model = train_model(train_df, "sales", model_type=model_type)
        metrics = evaluate_model(model, test_df, "sales")
        future = forecast_future(model, featured[["date", "sales"]], "date", "sales", horizon=horizon)

    st.success("Model trained and forecast generated ✅")

    # ---------------- KPI ROW ----------------
    st.subheader("📊 Model Performance (on held-out recent data)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE (avg. error)", f"{metrics['MAE']:,.1f}")
    k2.metric("RMSE", f"{metrics['RMSE']:,.1f}")
    k3.metric("MAPE", f"{metrics['MAPE (%)']:.1f}%")
    k4.metric("Approx. Accuracy", f"{metrics['Approx. Accuracy (%)']:.1f}%")

    st.caption(
        "MAE/RMSE are in the same units as your sales column (lower is better). "
        "Accuracy = 100% − average percentage error, a business-friendly way to read model performance."
    )

    # ---------------- CHART: HISTORY + TEST FIT + FUTURE FORECAST ----------------
    st.subheader("🔮 Forecast Visualization")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["sales"], name="Historical Sales",
        line=dict(color="#4C72B0", width=1.5), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=metrics["dates"], y=metrics["actuals"], name="Actual (recent test period)",
        line=dict(color="#2ca02c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=metrics["dates"], y=metrics["predictions"], name="Model Prediction (test period)",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=future["date"], y=future["sales"], name=f"Future Forecast (next {horizon} days)",
        line=dict(color="#d62728", width=3),
    ))
    fig.update_layout(
        template="plotly_white", height=480,
        xaxis_title="Date", yaxis_title="Sales / Demand",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- ERROR ANALYSIS ----------------
    st.subheader("🔍 Error Analysis")
    err_df = pd.DataFrame({
        "date": metrics["dates"],
        "actual": metrics["actuals"],
        "predicted": metrics["predictions"],
    })
    err_df["error"] = err_df["actual"] - err_df["predicted"]
    err_df["% error"] = (err_df["error"] / err_df["actual"].replace(0, pd.NA) * 100).round(1)

    c1, c2 = st.columns([2, 1])
    with c1:
        err_fig = go.Figure()
        err_fig.add_trace(go.Bar(x=err_df["date"], y=err_df["error"], marker_color="#8172B2"))
        err_fig.update_layout(
            template="plotly_white", height=320,
            title="Prediction Error Over Test Period (Actual − Predicted)",
            xaxis_title="Date", yaxis_title="Error",
            margin=dict(t=40, l=10, r=10, b=10),
        )
        st.plotly_chart(err_fig, use_container_width=True)
    with c2:
        st.markdown("**Recent test-period results**")
        st.dataframe(err_df.tail(10).round(1), use_container_width=True, hide_index=True)

    # ---------------- BUSINESS SUMMARY ----------------
    st.subheader("💼 Business-Ready Insight")
    st.info(business_summary(future, "sales", daily))

    st.markdown("**What this forecast means:** it projects expected daily sales/demand for the "
                "next period based on historical trend, weekly patterns, and seasonality.")
    st.markdown("**How a business can use it:** plan inventory levels, schedule staffing around peak "
                "demand days, manage cash flow expectations, and avoid overstocking during slow periods.")

    # ---------------- FORECAST TABLE + DOWNLOAD ----------------
    st.subheader("📋 Forecast Table")
    display_future = future.copy()
    display_future["sales"] = display_future["sales"].round(1)
    display_future.columns = ["Date", "Forecasted Sales"]
    st.dataframe(display_future, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Forecast as CSV",
        data=display_future.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.markdown("### 👋 Get started")
    st.markdown(
        """
        1. Pick a data source in the sidebar (sample dataset or your own CSV).
        2. Confirm which columns are the **date** and **sales/demand** values.
        3. Choose a forecasting method and horizon.
        4. Click **🚀 Run Forecast**.
        """
    )
    st.markdown("#### Preview of loaded data")
    st.dataframe(raw_df.head(15), use_container_width=True)
