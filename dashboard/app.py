import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# Configuration

st.set_page_config(

    page_title="Auditing LLM Trading",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="expanded"
)


# Data Paths

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"

RAW = DATA / "raw"

GT = DATA / "gt_tables"

PROCESSED = DATA / "processed"


# Helper Functions

@st.cache_data
def load_csv(path):

    return pd.read_csv(path)


def load_data():

    prices = load_csv(RAW / "market_prices.csv")

    predictions = load_csv(RAW / "llm_predictions.csv")

    gt = load_csv(GT / "gt_trade_table.csv")

    trust = load_csv(GT / "trust_scores.csv")

    return prices, predictions, gt, trust



# Sidebar

st.sidebar.title("📊 Auditing LLM Trading")

page = st.sidebar.radio(

    "Navigation",

    [

        "Overview",

        "Ground Truth",

        "Predictions",

        "Trust Score",

        "Performance",

        "Settings"

    ]

)


# Load Data

prices, predictions, gt_table, trust = load_data()


# Overview

if page == "Overview":

    st.title("📈 Auditing LLM Trading Dashboard")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Market Records",

        len(prices)

    )

    c2.metric(

        "Predictions",

        len(predictions)

    )

    c3.metric(

        "GT Records",

        len(gt_table)

    )

    c4.metric(

        "Average Trust",

        round(

            trust.TrustScore.mean(),

            2

        )

    )

    st.markdown("---")

    st.subheader("Price History")

    fig = px.line(

        prices,

        x="Date",

        y="Close",

        color="Ticker"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.subheader("Prediction Distribution")

    fig2 = px.histogram(

        predictions,

        x="Recommendation",

        color="Recommendation"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )


# GT TABLE

elif page == "Ground Truth":

    st.title("Ground Truth Table")

    st.dataframe(

        gt_table,

        use_container_width=True,

        height=600

    )

    st.download_button(

        "Download GT Table",

        gt_table.to_csv(index=False),

        file_name="gt_table.csv",

        mime="text/csv"

    )


# Predictions

elif page == "Predictions":

    st.title("LLM Predictions")

    ticker = st.selectbox(

        "Ticker",

        sorted(

            predictions.Ticker.unique()

        )

    )

    df = predictions[

        predictions.Ticker == ticker

    ]

    st.dataframe(df)

    if "Confidence" in df.columns:

        fig = px.line(

            df,

            x="Date",

            y="Confidence",

            title="Prediction Confidence"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )


# Trust Score

elif page == "Trust Score":

    st.title("Trust Score Analysis")

    st.dataframe(trust)

    fig = px.histogram(

        trust,

        x="TrustScore",

        nbins=25,

        title="Trust Score Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    fig = px.box(

        trust,

        y="TrustScore"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )


# Performance

elif page == "Performance":

    st.title("Model Performance")

    if "Correct" in gt_table.columns:

        accuracy = (

            gt_table.Correct.mean()

            * 100

        )

        st.metric(

            "Prediction Accuracy",

            f"{accuracy:.2f}%"

        )

    if "ActualReturn" in gt_table.columns:

        fig = px.histogram(

            gt_table,

            x="ActualReturn",

            nbins=40

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )


# Settings

elif page == "Settings":

    st.title("Dashboard Settings")

    dark = st.checkbox(

        "Dark Mode",

        value=False

    )

    refresh = st.slider(

        "Auto Refresh (seconds)",

        5,

        120,

        30

    )

    st.write("Dark Mode:", dark)

    st.write("Refresh:", refresh)


# Footer

st.markdown("---")

st.caption(

    "Auditing LLM Trading • Research Dashboard • Version 1.0"
)