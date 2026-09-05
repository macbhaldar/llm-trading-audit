"""
GT Table Visualizations
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

# Prediction Accuracy

def prediction_accuracy(
    self,
    df,
):
    correct = (
        df["Prediction"] ==
        df["GroundTruth"]
    ).sum()

    total = len(df)

    return round(
        correct / total,
        4,
    )

# Agreement Column

def add_agreement_column(
    self,
    df,
):

    data = df.copy()
    data["Agreement"] = np.where(
        data["Prediction"] ==
        data["GroundTruth"],
        "Correct",
        "Incorrect",
    )
    return data

# Accuracy by Model

def accuracy_by_model(
    self,
    df,
):
    data = self.add_agreement_column(df)
    result = (
        data
        .groupby("Model")["Agreement"]
        .apply(
            lambda x:
            (x == "Correct").mean()
        )
        .reset_index()
    )
    result.columns = [
        "Model",
        "Accuracy",
    ]
    return result


# Accuracy by Ticker

def accuracy_by_ticker(
    self,
    df,
):
    data = self.add_agreement_column(df)
    result = (
        data
        .groupby("Ticker")["Agreement"]
        .apply(
            lambda x:
            (x == "Correct").mean()
        )
        .reset_index()
    )
    result.columns = [
        "Ticker",
        "Accuracy",
    ]
    return result


# Confusion Matrix

def confusion_matrix(
    self,
    df,
):
    return pd.crosstab(
        df["GroundTruth"],
        df["Prediction"],
        margins=True,
    )


# GT vs Prediction Table

def comparison_table(
    self,
    df,
):
    data = self.add_agreement_column(df)
    return data[
        [
            "Date",
            "Ticker",
            "Model",
            "GroundTruth",
            "Prediction",
            "Agreement",
            "Confidence",
            "TrustScore",
        ]
    ]


# Incorrect Predictions

def incorrect_predictions(
    self,
    df,
):
    return df[
        df["Prediction"]
        !=
        df["GroundTruth"]
    ]


# Correct Predictions

def correct_predictions(
    self,
    df,
):
    return df[
        df["Prediction"]
        ==
        df["GroundTruth"]
    ]


# High Confidence Errors

def high_confidence_errors(
    self,
    df,
    threshold=0.90,
):
    return df[
        (
            df["Prediction"]
            !=
            df["GroundTruth"]
        )
        &
        (
            df["Confidence"]
            >= threshold
        )
    ]


# Low Trust Correct Predictions

def low_trust_correct(
    self,
    df,
    threshold=0.50,
):
    return df[
        (
            df["Prediction"]
            ==
            df["GroundTruth"]
        )
        &
        (
            df["TrustScore"]
            < threshold
        )
    ]


# Agreement Summary

def agreement_summary(
    self,
    df,
):
    correct = (
        df["Prediction"]
        ==
        df["GroundTruth"]
    ).sum()
    incorrect = len(df) - correct
    return {
        "Correct":
            int(correct),
        "Incorrect":
            int(incorrect),
        "Accuracy":
            round(
                correct /
                len(df),
                4,
            ),
    }


# Trust vs Agreement

def trust_accuracy_summary(
    self,
    df,
):
    data = self.add_agreement_column(df)
    return (
        data
        .groupby("Agreement")
        [
            [
                "TrustScore",
                "Confidence",
            ]
        ]
        .mean()
        .round(3)
    )


# Prediction Distribution

def prediction_distribution(
    self,
    df,
):
    return (
        df
        .groupby(
            "Prediction"
        )
        .size()
        .reset_index(
            name="Count"
        )
    )


# Ground Truth Distribution

def groundtruth_distribution(
    self,
    df,
):
    return (
        df
        .groupby(
            "GroundTruth"
        )
        .size()
        .reset_index(
            name="Count"
        )
    )


# Model Summary

def model_summary(
    self,
    df,
):
    result = (
        df
        .groupby("Model")
        .agg(
            Predictions=("Prediction", "count"),
            AvgConfidence=("Confidence", "mean"),
            AvgTrust=("TrustScore", "mean"),
        )
        .round(3)
    )
    return result.reset_index()


# Ticker Summary

def ticker_summary(
    self,
    df,
):
    result = (
        df
        .groupby("Ticker")
        .agg(
            Predictions=("Prediction", "count"),
            AvgConfidence=("Confidence", "mean"),
            AvgTrust=("TrustScore", "mean"),
        )
        .round(3)
    )
    return result.reset_index()


# Show Comparison Table

def show_comparison(
    self,
    df,
):
    st.subheader("Ground Truth vs Model Prediction")
    st.dataframe(
        self.comparison_table(df),
        use_container_width=True,
    )


# Show Confusion Matrix

def show_confusion_matrix(
    self,
    df,
):
    st.subheader("Confusion Matrix")
    st.dataframe(
        self.confusion_matrix(df),
        use_container_width=True,
    )

# Show Errors

def show_errors(
    self,
    df,
):
    st.subheader("Prediction Errors")
    st.dataframe(
        self.incorrect_predictions(df),
        use_container_width=True,
    )


# Model Leaderboard

def model_leaderboard(self, df):
    data = self.add_agreement_column(df)
    leaderboard = (
        data
        .groupby("Model")
        .agg(
            Accuracy=("Agreement",
                      lambda x: (x == "Correct").mean()),
            Trust=("TrustScore", "mean"),
            Confidence=("Confidence", "mean"),
            Trades=("Prediction", "count"),
        )
        .round(4)
    )

    leaderboard["Score"] = (
        0.50 * leaderboard["Accuracy"]
        + 0.30 * leaderboard["Trust"]
        + 0.20 * leaderboard["Confidence"]
    )
    leaderboard = (
        leaderboard
        .sort_values(
            "Score",
            ascending=False,
        )
        .reset_index()
    )
    leaderboard.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(leaderboard) + 1,
        ),
    )
    return leaderboard


# Asset Leaderboard

def asset_leaderboard(self, df):
    data = self.add_agreement_column(df)
    leaderboard = (
        data
        .groupby("Ticker")
        .agg(
            Accuracy=("Agreement",
                      lambda x: (x == "Correct").mean()),
            Trust=("TrustScore", "mean"),
            Confidence=("Confidence", "mean"),
            Predictions=("Prediction", "count"),
        )
        .round(4)
        .sort_values(
            "Accuracy",
            ascending=False,
        )
    )
    return leaderboard.reset_index()


# Trade Audit Table

def trade_audit_table(self, df):
    audit = df.copy()
    audit["Correct"] = (
        audit["Prediction"]
        ==
        audit["GroundTruth"]
    )
    audit["Error"] = (
        audit["Prediction"]
        !=
        audit["GroundTruth"]
    )
    audit["RiskFlag"] = np.where(
        (
            audit["Confidence"] > 0.90
        )
        &
        (
            audit["Error"]
        ),
        "HIGH",
        "NORMAL",
    )
    return audit


# GT Scorecard

def gt_scorecard(self, df):
    accuracy = self.prediction_accuracy(df)
    trust = df["TrustScore"].mean()
    confidence = df["Confidence"].mean()
    hallucination = (
        df["Prediction"]
        !=
        df["GroundTruth"]
    ).mean()
    return {
        "Accuracy":
            round(accuracy, 4),
        "Trust":
            round(trust, 4),
        "Confidence":
            round(confidence, 4),
        "Hallucination Rate":
            round(hallucination, 4),
    }


# Model Ranking

def rank_models(self, df):
    board = self.model_leaderboard(df)
    return board.sort_values(
        "Score",
        ascending=False,
    )

# Trust Ranking

def trust_ranking(self, df):
    return (
        df
        .groupby("Model")["TrustScore"]
        .mean()
        .sort_values(
            ascending=False,
        )
        .reset_index()
    )


# Confidence Ranking

def confidence_ranking(self, df):
    return (
        df
        .groupby("Model")["Confidence"]
        .mean()
        .sort_values(
            ascending=False,
        )
        .reset_index()
    )


# Hallucination Ranking

def hallucination_ranking(self, df):
    data = self.add_agreement_column(df)
    ranking = (
        data
        .groupby("Model")["Agreement"]
        .apply(
            lambda x:
            (x == "Incorrect").mean()
        )
        .reset_index()
    )
    ranking.columns = [
        "Model",
        "HallucinationRate",
    ]
    return ranking.sort_values(
        "HallucinationRate"
    )

# Calibration Ranking

def calibration_ranking(self, df):
    ranking = (
        df
        .groupby("Model")
        .apply(
            lambda x:
            np.mean(
                np.abs(
                    x["Confidence"]
                    -
                    x["TrustScore"]
                )
            )
        )
        .reset_index()
    )
    ranking.columns = [
        "Model",
        "CalibrationError",
    ]
    return ranking.sort_values(
        "CalibrationError"
    )

# Overall Audit Report

def audit_report(self, df):
    return {
        "Scorecard":
            self.gt_scorecard(df),
        "Leaderboard":
            self.model_leaderboard(df),
        "Assets":
            self.asset_leaderboard(df),
        "Hallucination":
            self.hallucination_ranking(df),
        "Calibration":
            self.calibration_ranking(df),
    }

# Executive Summary

def executive_summary(self, df):
    score = self.gt_scorecard(df)
    return pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Trust",
                "Confidence",
                "Hallucination",
            ],
            "Value": [
                score["Accuracy"],
                score["Trust"],
                score["Confidence"],
                score["Hallucination Rate"],
            ],
        }
    )

# Export Report

def export_audit_report(
    self,
    df,
    filename="audit_report.csv",
):
    report = self.executive_summary(df)
    st.download_button(
        "Download Audit Report",
        report.to_csv(index=False),
        filename,
        mime="text/csv",
    )

# Show Leaderboard

def show_leaderboard(self, df):
    st.subheader("Model Leaderboard")
    st.dataframe(
        self.model_leaderboard(df),
        use_container_width=True,
    )

# Show Asset Leaderboard

def show_assets(self, df):
    st.subheader("Asset Leaderboard")
    st.dataframe(
        self.asset_leaderboard(df),
        use_container_width=True,
    )

# Show Executive Summary

def show_summary_report(self, df):
    st.subheader("Executive Summary")
    st.dataframe(
        self.executive_summary(df),
        use_container_width=True,
    )

# Full Audit Dashboard

def render_audit_dashboard(self, df):
    self.show_summary(df)
    st.divider()
    self.show_leaderboard(df)
    st.divider()
    self.show_assets(df)
    st.divider()
    self.show_summary_report(df)
    st.divider()
    self.export_audit_report(df)


# Rolling Trust Score

def rolling_trust(
    self,
    df,
    window=30,
):
    data = df.copy()
    data = data.sort_values("Date")
    data["RollingTrust"] = (
        data["TrustScore"]
        .rolling(window)
        .mean()
    )
    return data


# Rolling Accuracy

def rolling_accuracy(
    self,
    df,
    window=30,
):
    data = self.add_agreement_column(df)
    data["Correct"] = (
        data["Agreement"]
        ==
        "Correct"
    ).astype(int)
    data = data.sort_values("Date")
    data["RollingAccuracy"] = (
        data["Correct"]
        .rolling(window)
        .mean()
    )
    return data


# Daily Audit Summary

def daily_summary(
    self,
    df,
):
    data = self.add_agreement_column(df)
    summary = (
        data
        .groupby("Date")
        .agg(
            Accuracy=(
                "Agreement",
                lambda x:
                (x == "Correct").mean(),
            ),
            Trust=(
                "TrustScore",
                "mean",
            ),
            Confidence=(
                "Confidence",
                "mean",
            ),
            Trades=(
                "Prediction",
                "count",
            ),
        )
        .round(4)
    )
    return summary.reset_index()


# Monthly Audit Summary

def monthly_summary(
    self,
    df,
):
    data = df.copy()
    data["Date"] = pd.to_datetime(
        data["Date"]
    )
    data["Month"] = (
        data["Date"]
        .dt.to_period("M")
        .astype(str)
    )
    data = self.add_agreement_column(data)
    summary = (
        data
        .groupby("Month")
        .agg(
            Accuracy=(
                "Agreement",
                lambda x:
                (x == "Correct").mean(),
            ),
            Trust=(
                "TrustScore",
                "mean",
            ),
            Confidence=(
                "Confidence",
                "mean",
            ),
            Trades=(
                "Prediction",
                "count",
            ),
        )
        .round(4)
    )
    return summary.reset_index()


# Trust Heatmap

def trust_heatmap(
    self,
    df,
):
    table = pd.pivot_table(
        df,
        index="Model",
        columns="Ticker",
        values="TrustScore",
        aggfunc=np.mean,
    )
    return table.round(3)

# Accuracy Heatmap

def accuracy_heatmap(
    self,
    df,
):
    data = self.add_agreement_column(df)
    data["Correct"] = (
        data["Agreement"]
        ==
        "Correct"
    ).astype(int)

    table = pd.pivot_table(
        data,
        index="Model",
        columns="Ticker",
        values="Correct",
        aggfunc=np.mean,
    )
    return table.round(3)


# Model Drift

def model_drift(
    self,
    df,
):
    data = df.copy()
    data["Date"] = pd.to_datetime(
        data["Date"]
    )
    data["Month"] = (
        data["Date"]
        .dt.to_period("M")
        .astype(str)
    )
    drift = (
        data
        .groupby(
            [
                "Month",
                "Model",
            ]
        )
        ["TrustScore"]
        .mean()
        .reset_index()
    )
    return drift


# Drift Score

def drift_score(
    self,
    df,
):
    drift = self.model_drift(df)
    result = (
        drift
        .groupby("Model")
        ["TrustScore"]
        .std()
        .reset_index()
    )
    result.columns = [
        "Model",
        "DriftScore",
    ]
    return result.sort_values(
        "DriftScore",
        ascending=False,
    )


# Stable Models

def stable_models(
    self,
    df,
):
    drift = self.drift_score(df)
    return drift.sort_values(
        "DriftScore",
        ascending=True,
    )


# Trust Change

def trust_change(
    self,
    df,
):
    monthly = self.monthly_summary(df)
    monthly["TrustChange"] = (
        monthly["Trust"]
        .diff()
    )
    return monthly


# Accuracy Change

def accuracy_change(
    self,
    df,
):
    monthly = self.monthly_summary(df)
    monthly["AccuracyChange"] = (
        monthly["Accuracy"]
        .diff()
    )
    return monthly


# Risk Flags

def temporal_risk_flags(
    self,
    df,
):
    monthly = self.monthly_summary(df)
    monthly["Risk"] = np.where(
        (
            monthly["Trust"] < 0.60
        )
        |
        (
            monthly["Accuracy"] < 0.55
        ),
        "HIGH",
        "NORMAL",
    )
    return monthly


# Audit Timeline

def audit_timeline(
    self,
    df,
):
    daily = self.daily_summary(df)
    return daily[
        [
            "Date",
            "Accuracy",
            "Trust",
            "Confidence",
            "Trades",
        ]
    ]


# Time Window Comparison

def compare_periods(
    self,
    df,
    start1,
    end1,
    start2,
    end2,
):
    p1 = df[
        (
            df["Date"] >= start1
        )
        &
        (
            df["Date"] <= end1
        )
    ]
    p2 = df[
        (
            df["Date"] >= start2
        )
        &
        (
            df["Date"] <= end2
        )
    ]
    return {
        "Period1":
            self.gt_scorecard(p1),
        "Period2":
            self.gt_scorecard(p2),
    }


# Trust Trend by Model

def trust_trend_by_model(
    self,
    df,
):
    data = df.copy()
    data["Date"] = pd.to_datetime(
        data["Date"]
    )
    data["Month"] = (
        data["Date"]
        .dt.to_period("M")
        .astype(str)
    )
    return (
        data
        .groupby(
            [
                "Month",
                "Model",
            ]
        )
        ["TrustScore"]
        .mean()
        .reset_index()
    )


# Prediction Flow Table

def prediction_flow(
    self,
    df,
):
    """
    Creates a flow table betweenGround Truth -> Prediction.
    """
    flow = (
        df
        .groupby(
            [
                "GroundTruth",
                "Prediction",
            ]
        )
        .size()
        .reset_index(name="Count")
        .sort_values(
            "Count",
            ascending=False,
        )
    )
    return flow


# Model Prediction Counts

def model_prediction_counts(
    self,
    df,
):
    counts = (
        df
        .groupby(
            [
                "Model",
                "Prediction",
            ]
        )
        .size()
        .reset_index(name="Count")
    )
    return counts


# Ground Truth Counts

def groundtruth_counts(
    self,
    df,
):
    counts = (
        df
        .groupby("GroundTruth")
        .size()
        .reset_index(name="Count")
    )
    return counts
