import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Buyer Dashboard", layout="wide")

st.title("Amazon Smart Buyer Analytics Dashboard")

# ----------------------------
# LOAD DATA
# ----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/amazon_master_dataset.csv")
    return df

df = load_data()

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------

st.sidebar.header("Filters")

brand = st.sidebar.multiselect(
    "Brand",
    df["brand"].dropna().unique(),
    default=df["brand"].dropna().unique()
)

subcategory = st.sidebar.multiselect(
    "Subcategory",
    df["subcategory"].dropna().unique(),
    default=df["subcategory"].dropna().unique()
)

filtered = df[
    (df["brand"].isin(brand)) &
    (df["subcategory"].isin(subcategory))
]

# ==================================================
# 1️⃣ Margin Distribution
# ==================================================

st.subheader("Margin Distribution")

margin_fig = px.histogram(
    filtered,
    x="MarginPercent",
    nbins=40,
)

margin_fig.update_traces(marker_color="teal")

st.plotly_chart(margin_fig, use_container_width=True)

# ==================================================
# 2️⃣ Category Momentum Distribution
# ==================================================

st.subheader("Category Momentum Distribution")

momentum_fig = px.histogram(
    filtered,
    x="CategoryMomentum",
    nbins=40
)

momentum_fig.update_traces(marker_color="green")

st.plotly_chart(momentum_fig, use_container_width=True)

# ==================================================
# 3️⃣ Momentum vs Drops Per Day
# ==================================================

st.subheader("Momentum vs Drops Per Day")

scatter_fig = px.scatter(
    filtered,
    x="CatDropsPerDay",
    y="CategoryMomentum",
    color="Demand_Tier",
    hover_data=["subcategory","brand"],
)

st.plotly_chart(scatter_fig, use_container_width=True)

# ==================================================
# 4️⃣ Demand Tier Confusion Matrix
# ==================================================

st.subheader("Demand Tier Confusion Matrix")

# Only works if these columns exist
if {"target_tier","pred_class"}.issubset(filtered.columns):

    confusion = pd.crosstab(
        filtered["target_tier"],
        filtered["pred_class"]
    )

    heatmap = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="Blues"
    )

    st.plotly_chart(heatmap, use_container_width=True)

else:
    st.info("Confusion matrix requires 'target_tier' and 'pred_class' columns.")

# ==================================================
# 5️⃣ OOS Rate by Brand
# ==================================================

st.subheader("Out-of-Stock Rate by Brand")

if "oos_continuous" in filtered.columns:

    oos_brand = (
        filtered.groupby("brand")["oos_continuous"]
        .mean()
        .reset_index()
        .sort_values("oos_continuous", ascending=False)
    )

    oos_chart = px.bar(
        oos_brand,
        x="oos_continuous",
        y="brand",
        orientation="h",
        color="oos_continuous"
    )

    st.plotly_chart(oos_chart, use_container_width=True)

else:
    st.info("OOS analysis requires 'oos_continuous' column.")

# ==================================================
# 6️⃣ OOS Feature Importance
# ==================================================

st.subheader("OOS Feature Importance")

try:

    importance = pd.read_csv("data/oos_feature_importance.csv")

    imp_chart = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h"
    )

    imp_chart.update_traces(marker_color="#4c78a8")

    st.plotly_chart(imp_chart, use_container_width=True)

except:
    st.info("Upload 'oos_feature_importance.csv' to data folder.")

# ==================================================
# 7️⃣ Forecasted Competitive Intensity
# ==================================================

st.subheader("Forecasted Competitive Intensity")

try:

    forecast = pd.read_csv("data/cii_forecast_2026.csv")

    forecast_chart = px.line(
        forecast,
        x="quarter",
        y="CII_Score",
        color="subcategory"
    )

    st.plotly_chart(forecast_chart, use_container_width=True)

except:
    st.info("Upload 'cii_forecast_2026.csv' to data folder.")

# ==================================================
# PRODUCT TABLE
# ==================================================

st.subheader("Product Explorer")

st.dataframe(
    filtered[
        [
            "asin",
            "title",
            "brand",
            "subcategory",
            "MarginPercent",
            "CategoryMomentum",
            "CatDropsPerDay",
            "Demand_Tier",
            "CII_Score"
        ]
    ]
)
