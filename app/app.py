import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from src.data_preprocessing import load_data
from src.feature_engineering import FEATURES

st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="📦",
    layout="wide"
)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    return joblib.load("models/demand_model.pkl")


# Load data and model
df = get_data()
model = get_model()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("📦 Supply Chain Demand Forecasting & Inventory Optimization")
st.caption(
    "Placement-ready Data Science dashboard: "
    "demand analytics, forecasting and inventory recommendations."
)


# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------
total_sales = int(df["sales"].sum())
avg_daily = df.groupby("date")["sales"].sum().mean()
latest_date = df["date"].max()

if Path("models/metrics.json").exists():
    metrics = pd.read_json("models/metrics.json", typ="series")
    mape = float(metrics["MAPE"])
else:
    mape = 0

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Units Sold",
    f"{total_sales:,}"
)

c2.metric(
    "Avg Daily Demand",
    f"{avg_daily:,.0f}"
)

c3.metric(
    "Stores",
    df["store"].nunique()
)

c4.metric(
    "Items",
    df["item"].nunique()
)

st.divider()


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📈 Sales Analytics",
        "🔮 Demand Forecast",
        "📦 Inventory"
    ]
)


# =========================================================
# TAB 1 - SALES ANALYTICS
# =========================================================
with tab1:

    st.subheader("Sales Trend")

    daily = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
    )

    fig = px.line(
        daily,
        x="date",
        y="sales",
        title="Daily Total Sales"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


    left, right = st.columns(2)


    # Store Sales
    with left:

        store_sales = (
            df.groupby("store", as_index=False)["sales"]
            .sum()
        )

        fig2 = px.bar(
            store_sales,
            x="store",
            y="sales",
            title="Sales by Store"
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )


    # Item Sales
    with right:

        item_sales = (
            df.groupby("item", as_index=False)["sales"]
            .sum()
            .sort_values(
                "sales",
                ascending=False
            )
            .head(10)
        )

        fig3 = px.bar(
            item_sales,
            x="item",
            y="sales",
            title="Top 10 Items"
        )

        st.plotly_chart(
            fig3,
            width="stretch"
        )


# =========================================================
# TAB 2 - DEMAND FORECAST
# =========================================================
with tab2:

    st.subheader("30-Day Demand Forecast")


    # Store selection
    store = st.selectbox(
        "Select Store",
        sorted(df["store"].unique())
    )


    # Item selection
    item = st.selectbox(
        "Select Item",
        sorted(df["item"].unique())
    )


    # Filter historical data
    history = (
        df[
            (df["store"] == store)
            & (df["item"] == item)
        ]
        .sort_values("date")
        .copy()
    )


    st.write(
        f"Latest historical date: "
        f"**{history['date'].max().date()}**"
    )


    # -----------------------------------------------------
    # RECURSIVE 30-DAY FORECAST
    # -----------------------------------------------------

    series = history[
        ["date", "sales"]
    ].copy()

    values = list(
        series["sales"]
        .astype(float)
        .values
    )


    future_dates = pd.date_range(
        history["date"].max()
        + pd.Timedelta(days=1),
        periods=30
    )


    predictions = []


    for dt in future_dates:

        if len(values) < 28:
            break


        row = pd.DataFrame(
            [
                {
                    "store": store,
                    "item": item,

                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,

                    "day_of_week": dt.day_of_week,

                    "is_weekend": int(
                        dt.day_of_week >= 5
                    ),

                    "day_of_year": dt.day_of_year,

                    "week_of_year": int(
                        dt.isocalendar().week
                    ),

                    "lag_1": values[-1],
                    "lag_7": values[-7],
                    "lag_14": values[-14],
                    "lag_28": values[-28],

                    "rolling_7": np.mean(
                        values[-7:]
                    ),

                    "rolling_14": np.mean(
                        values[-14:]
                    ),

                    "rolling_28": np.mean(
                        values[-28:]
                    )
                }
            ]
        )


        # Model prediction
        pred = max(
            0,
            float(
                model.predict(
                    row[FEATURES]
                )[0]
            )
        )


        predictions.append(pred)

        # Add prediction for recursive forecasting
        values.append(pred)


    # -----------------------------------------------------
    # FORECAST DATAFRAME
    # -----------------------------------------------------

    forecast = pd.DataFrame(
        {
            "date": future_dates[
                :len(predictions)
            ],

            "sales": predictions
        }
    )


    # -----------------------------------------------------
    # DISPLAY FORECAST TABLE
    # -----------------------------------------------------

    if forecast.empty:

        st.error(
            "Unable to generate forecast."
        )

    else:

        st.subheader(
            "Forecasted Daily Demand"
        )

        forecast_display = forecast.copy()

        forecast_display["date"] = (
            forecast_display["date"]
            .dt.date
        )

        forecast_display["sales"] = (
            forecast_display["sales"]
            .round(2)
        )

        st.dataframe(
            forecast_display,
            width="stretch",
            hide_index=True
        )


        # -------------------------------------------------
        # HISTORICAL + FORECAST CHART
        # -------------------------------------------------

        hist_tail = (
            history
            .tail(90)
            [["date", "sales"]]
            .rename(
                columns={
                    "sales": "value"
                }
            )
        )


        fc_plot = (
            forecast
            .rename(
                columns={
                    "sales": "value"
                }
            )
        )


        hist_tail["type"] = "Historical"
        fc_plot["type"] = "Forecast"


        chart_df = pd.concat(
            [
                hist_tail,
                fc_plot
            ],
            ignore_index=True
        )


        fig = px.line(
            chart_df,
            x="date",
            y="value",
            color="type",
            title=(
                f"Store {store} / "
                f"Item {item}"
            )
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


        # -------------------------------------------------
        # TOTAL FORECAST
        # -------------------------------------------------

        total_forecast = (
            forecast["sales"].sum()
        )


        st.metric(
            "Forecasted 30-Day Demand",
            f"{total_forecast:,.0f} units"
        )


# =========================================================
# TAB 3 - INVENTORY
# =========================================================
with tab3:

    st.subheader(
        "Inventory Recommendations"
    )


    rec_path = Path(
        "data/processed/"
        "inventory_recommendations.csv"
    )


    if not rec_path.exists():

        st.warning(
            "Run "
            "`python -m src.build_outputs` "
            "first."
        )

    else:

        rec = pd.read_csv(
            rec_path
        )


        # -------------------------------------------------
        # INVENTORY STATUS FILTER
        # -------------------------------------------------

        statuses = sorted(
            rec["status"]
            .dropna()
            .unique()
        )


        status = st.multiselect(
            "Inventory Status",
            statuses,
            default=statuses
        )


        view = (
            rec[
                rec["status"].isin(status)
            ]
            .sort_values(
                "recommended_order",
                ascending=False
            )
        )


        # -------------------------------------------------
        # INVENTORY TABLE
        # -------------------------------------------------

        st.subheader(
            "Inventory Recommendation Table"
        )


        st.dataframe(
            view,
            width="stretch",
            hide_index=True
        )


        # -------------------------------------------------
        # DOWNLOAD CSV
        # -------------------------------------------------

        st.download_button(
            "Download Recommendations CSV",
            view.to_csv(
                index=False
            ),
            "inventory_recommendations.csv",
            "text/csv"
        )