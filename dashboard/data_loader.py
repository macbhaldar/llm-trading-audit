from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from dashboard.config import REQUIRED_FILES


class DashboardDataLoader:
    """
    Loads, validates, caches and summarizes dashboard datasets.
    """

    def __init__(self):
        self.files = REQUIRED_FILES

    # Generic CSV Loader
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_csv(path: Path) -> pd.DataFrame:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{path}"
            )

        return pd.read_csv(path)

    # Individual Loaders
    def load_market_prices(self):
        return self.load_csv(
            self.files["prices"]
        )

    def load_predictions(self):
        return self.load_csv(
            self.files["predictions"]
        )

    def load_gt_table(self):
        return self.load_csv(
            self.files["gt"]
        )

    def load_trust_scores(self):
        return self.load_csv(
            self.files["trust"]
        )

    def load_backtest(self):
        return self.load_csv(
            self.files["backtest"]
        )

    def load_metrics(self):
        return self.load_csv(
            self.files["metrics"]
        )
    
    # Load Everything
    def load_all(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for key, path in self.files.items():
            try:
                data[key] = self.load_csv(path)
            except FileNotFoundError:
                data[key] = pd.DataFrame()
        return data

    # Validation
    @staticmethod
    def validate(df: pd.DataFrame):
        return {
            "Rows": len(df),
            "Columns": len(df.columns),
            "Duplicates": int(
                df.duplicated().sum()
            ),
            "Missing Values":
                df.isna().sum().to_dict(),
        }

    # Convert Dates
    @staticmethod
    def parse_dates(
        df: pd.DataFrame,
        column="Date",
    ):

        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )
        return df

    # Sort Dataset
    @staticmethod
    def sort(
        df,
        by="Date"
    ):

        if by in df.columns:
            return df.sort_values(by)
        return df

    # Filter by ticker
    @staticmethod
    def filter_ticker(
        df,
        ticker,
    ):

        if "Ticker" not in df.columns:
            return df
        return df[
            df["Ticker"] == ticker
        ]

    # Filter Date Range
    @staticmethod
    def filter_date(
        df,
        start,
        end,
    ):

        if "Date" not in df.columns:
            return df
        return df[
            (df["Date"] >= start)
            &
            (df["Date"] <= end)
        ]

    # Summary Statistics
    @staticmethod
    def summary(df):
        return {
            "Rows":
                len(df),

            "Columns":
                len(df.columns),

            "Numeric Columns":
                len(
                    df.select_dtypes(
                        "number"
                    ).columns
                ),

            "Categorical Columns":

                len(
                    df.select_dtypes(
                        exclude="number"
                    ).columns
                ),
        }

    # Numeric Statistics 
    @staticmethod
    def describe(df):
        return df.describe().T

    # Column Names
    @staticmethod
    def columns(df):
        return list(df.columns)

    # Unique Tickers
    @staticmethod
    def tickers(df):
        if "Ticker" not in df.columns:
            return []
        return sorted(
            df["Ticker"]
            .dropna()
            .unique()
        )

    # Export
    @staticmethod
    def to_csv(df):
        return df.to_csv(
            index=False
        ).encode("utf-8")


# Convenience Function
@st.cache_resource
def get_loader():
    return DashboardDataLoader()