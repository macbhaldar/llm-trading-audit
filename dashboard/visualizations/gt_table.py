"""
GT Table Visualizations (Part 1) : Base GTTableVisualizer class, interactive GT table, 
filtering, searching, trust highlighting, export.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import streamlit as st
from .base import BaseVisualizer


class GTTableVisualizer(BaseVisualizer):
    def __init__(self):
        super().__init__()

    
    # Required Columns
    
    REQUIRED_COLUMNS = [
        "Date",
        "Ticker",
        "Model",
        "Prediction",
        "GroundTruth",
        "Confidence",
        "TrustScore",
    ]

    
    # Validate GT Table
    
    def validate_gt_table(
        self,
        df: pd.DataFrame,
    ):
        self.validate_columns(
            df,
            self.REQUIRED_COLUMNS,
        )
        return True
    
    # Search
    
    def search(
        self,
        df,
        keyword,
    ):
        if keyword is None or keyword == "":
            return df
        mask = np.column_stack([
            df[col]
            .astype(str)
            .str.contains(
                keyword,
                case=False,
                na=False,
            )
            for col in df.columns
        ])
        return df.loc[mask.any(axis=1)]
    
    # Filter Model
    
    def filter_model(
        self,
        df,
        model,
    ):
        if model == "All":
            return df
        return df[
            df["Model"] == model
        ]

    
    # Filter Ticker
    
    def filter_ticker(
        self,
        df,
        ticker,
    ):
        if ticker == "All":
            return df
        return df[
            df["Ticker"] == ticker
        ]

    
    # Filter Trust
    
    def filter_trust(
        self,
        df,
        minimum=0.0,
        maximum=1.0,
    ):
        return df[
            (df["TrustScore"] >= minimum)
            &
            (df["TrustScore"] <= maximum)
        ]

    
    # Filter Date
    
    def filter_date(
        self,
        df,
        start_date,
        end_date,
    ):

        return df[
            (df["Date"] >= start_date)
            &
            (df["Date"] <= end_date)
        ]
    
    # Highlight Trust
    
    @staticmethod
    def highlight_trust(value):
        if value >= 0.80:
            return "background-color:#D1FAE5"
        elif value >= 0.60:
            return "background-color:#FEF3C7"
        else:
            return "background-color:#FECACA"

    
    # Highlight Prediction

    @staticmethod
    def highlight_prediction(row):
        if row["Prediction"] == row["GroundTruth"]:
            return [
                "background-color:#DCFCE7"
            ] * len(row)
        else:
            return [
                "background-color:#FEE2E2"
            ] * len(row)

    
    # Summary
    
    def summary(
        self,
        df,
    ):
        return {
            "Rows":
                len(df),

            "Unique Models":
                df["Model"].nunique(),

            "Tickers":
                df["Ticker"].nunique(),

            "Average Trust":
                round(
                    df["TrustScore"].mean(),
                    3,
                ),

            "Average Confidence":
                round(
                    df["Confidence"].mean(),
                    3,
                ),
        }
    
    # Display Summary

    def show_summary(
        self,
        df,
    ):
        summary = self.summary(df)

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Rows",
            summary["Rows"],
        )

        c2.metric(
            "Models",
            summary["Unique Models"],
        )

        c3.metric(
            "Assets",
            summary["Tickers"],
        )

        c4.metric(
            "Avg Trust",
            summary["Average Trust"],
        )

        c5.metric(
            "Avg Confidence",
            summary["Average Confidence"],
        )

    
    # Interactive GT Table
    
    def show_table(
        self,
        df,
        height=650,
    ):

        styled = (
            df.style
            .apply(
                self.highlight_prediction,
                axis=1,
            )

            .map(
                self.highlight_trust,
                subset=["TrustScore"],
            )
        )

        st.dataframe(
            styled,
            use_container_width=True,
            height=height,
        )
    
    # Download
    
    def download_button(
        self,
        df,
        filename="gt_table.csv",
    ):
        st.download_button(
            label="Download GT Table",
            data=df.to_csv(index=False),
            file_name=filename,
            mime="text/csv",
        )
    
    # Sidebar Filters

    def sidebar_filters(
        self,
        df,
    ):

        st.sidebar.header("GT Table Filters")

        models = [
            "All"
        ] + sorted(
            df["Model"].unique()
        )

        tickers = [
            "All"
        ] + sorted(
            df["Ticker"].unique()
        )

        model = st.sidebar.selectbox(
            "Model",
            models,
        )

        ticker = st.sidebar.selectbox(
            "Ticker",
            tickers,
        )

        trust = st.sidebar.slider(

            "Minimum Trust",
            0.0,
            1.0,
            0.50,
            0.01,
        )

        keyword = st.sidebar.text_input(
            "Search",
            "",
        )

        data = self.filter_model(
            df,
            model,
        )

        data = self.filter_ticker(
            data,
            ticker,
        )

        data = self.filter_trust(
            data,
            trust,
            1.0,
        )

        data = self.search(
            data,
            keyword,
        )
        return data

    
    # Render
    
    def render(
        self,
        df,
    ):
        self.validate_gt_table(df)
        filtered = self.sidebar_filters(df)
        self.show_summary(filtered)
        st.divider()
        self.show_table(filtered)
        self.download_button(filtered)
