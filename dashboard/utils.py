from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st


# Formatting Utilities

def format_currency(value, symbol="$"):
    """
    Format numeric value as currency.
    """
    if pd.isna(value):
        return "N/A"
    return f"{symbol}{value:,.2f}"


def format_percentage(value):
    """
    Convert decimal to percentage string.
    """
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_number(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


# Date Utilities

def convert_date(df, column="Date"):
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )
    return df


def get_date_range(df, column="Date"):
    if column not in df.columns:
        return None, None
    return df[column].min(), df[column].max()


# Data Filtering

def filter_ticker(df, ticker):
    if "Ticker" not in df.columns:
        return df
    return df[df["Ticker"] == ticker]


def filter_date_range(
    df,
    start_date,
    end_date,
    column="Date",
):

    if column not in df.columns:
        return df

    return df[
        (df[column] >= start_date)
        &
        (df[column] <= end_date)
    ]


# Missing Values

def missing_summary(df):
    return pd.DataFrame(
        {
            "Missing":
            df.isna().sum(),
            "Percent":
            (
                df.isna().mean()
                * 100
            ).round(2),
        }
    )


# Dataset Information

def dataset_info(df):
    return {
        "Rows":
        len(df),
        "Columns":
        len(df.columns),
        "Memory(MB)":
        round(
            df.memory_usage(
                deep=True
            ).sum()
            / 1024**2,
            2,
        ),
    }


# Download Button

def download_dataframe(
    df,
    filename,
    label="Download CSV",
):

    st.download_button(
        label,
        df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
    )

# Status Messages

def success(message):
    st.success(message)

def warning(message):
    st.warning(message)


def error(message):
    st.error(message)


def info(message):
    st.info(message)


# Metrics

def metric_card(
    label,
    value,
    delta=None,
):

    st.metric(
        label,
        value,
        delta=delta,
    )


# File Utilities

def ensure_directory(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )

def file_exists(path):
    return Path(path).exists()


# Export

def export_excel(
    df,
    filename,
):
    return df.to_excel(
        filename,
        index=False,
    )


# Statistics

def summary_statistics(df):
    return df.describe().T

def correlation_matrix(df):
    return (
        df
        .select_dtypes(
            include=np.number
        )
        .corr()
    )



# Sidebar Filters

def sidebar_ticker_filter(df):
    tickers = sorted(
        df["Ticker"]
        .dropna()
        .unique()
    )

    return st.sidebar.selectbox(
        "Ticker",
        tickers,
    )


def sidebar_date_filter(df):
    start, end = get_date_range(df)
    return st.sidebar.date_input(
        "Date Range",
        (
            start,
            end,
        ),
    )

# Timestamp

def current_timestamp():
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# Page Header

def page_header(
    title,
    subtitle=None,
):
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()


# Data Validation

def check_required_columns(
    df,
    columns,
):
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]
    return missing
