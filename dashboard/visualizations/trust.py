"""
Trust Score Visualizations
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .base import BaseVisualizer


class TrustVisualizer(BaseVisualizer):
    # Trust Score Trend
    def trust_trend(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
        trust_col: str = "TrustScore",
        title: str = "Trust Score Trend",
    ):

        self.validate_columns(df, [date_col, trust_col])

        fig = px.line(
            df,
            x=date_col,
            y=trust_col,
            markers=True,
        )

        fig.update_traces(
            line=dict(width=3)
        )

        return self.apply_layout(fig, title)

    # Trust Distribution
    
    def trust_distribution(
        self,
        df,
        trust_col="TrustScore",
        title="Trust Score Distribution",
    ):

        self.validate_columns(df, [trust_col])

        fig = px.histogram(
            df,
            x=trust_col,
            nbins=30,
        )

        return self.apply_layout(fig, title)

    # Trust Boxplot
    
    def trust_boxplot(
        self,
        df,
        trust_col="TrustScore",
        title="Trust Score Boxplot",
    ):

        self.validate_columns(df, [trust_col])

        fig = px.box(
            df,
            y=trust_col,
            points="all",
        )

        return self.apply_layout(fig, title)

    # Trust Gauge
    
    def trust_gauge(
        self,
        score: float,
        title="Average Trust Score",
    ):

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": title},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": self.primary},
                    "steps": [
                        {
                            "range": [0, 0.40],
                            "color": "#ef4444",
                        },
                        {
                            "range": [0.40, 0.70],
                            "color": "#f59e0b",
                        },
                        {
                            "range": [0.70, 1.00],
                            "color": "#10b981",
                        },
                    ],
                },
            )
        )

        return self.apply_layout(fig, title, 400)

    # Trust by Model
    
    def trust_by_model(
        self,
        df,
        model_col="Model",
        trust_col="TrustScore",
        title="Trust Score by Model",
    ):

        self.validate_columns(
            df,
            [model_col, trust_col],
        )

        fig = px.box(
            df,
            x=model_col,
            y=trust_col,
            color=model_col,
        )

        return self.apply_layout(fig, title)

    # Trust by Ticker
    
    def trust_by_ticker(
        self,
        df,
        ticker_col="Ticker",
        trust_col="TrustScore",
        title="Trust Score by Asset",
    ):

        self.validate_columns(
            df,
            [ticker_col, trust_col],
        )

        summary = (
            df.groupby(ticker_col)[trust_col]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            summary,
            x=ticker_col,
            y=trust_col,
        )

        return self.apply_layout(fig, title)

    # Confidence vs Trust
    
    def confidence_vs_trust(
        self,
        df,
        confidence_col="Confidence",
        trust_col="TrustScore",
        color_col: Optional[str] = None,
        title="Confidence vs Trust",
    ):

        self.validate_columns(
            df,
            [confidence_col, trust_col],
        )

        fig = px.scatter(
            df,
            x=confidence_col,
            y=trust_col,
            color=color_col,
            trendline="ols",
        )

        return self.apply_layout(fig, title)

    # Rolling Trust
    
    def rolling_trust(
        self,
        df,
        trust_col="TrustScore",
        date_col="Date",
        window=20,
        title="Rolling Trust Score",
    ):

        self.validate_columns(
            df,
            [date_col, trust_col],
        )

        data = df.copy()

        data["RollingTrust"] = (
            data[trust_col]
            .rolling(window)
            .mean()
        )

        fig = px.line(
            data,
            x=date_col,
            y="RollingTrust",
        )

        return self.apply_layout(fig, title)

    # Trust Heatmap
    
    def trust_heatmap(
        self,
        df,
        index="Model",
        columns="Ticker",
        values="TrustScore",
        title="Trust Heatmap",
    ):

        self.validate_columns(
            df,
            [index, columns, values],
        )

        table = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=np.mean,
        )

        fig = px.imshow(
            table,
            text_auto=".2f",
            aspect="auto",
        )

        return self.apply_layout(fig, title)

    # Trust KPI
    
    def trust_summary(
        self,
        df,
        trust_col="TrustScore",
    ):

        self.validate_columns(df, [trust_col])

        values = df[trust_col]

        return {
            "Mean": float(values.mean()),
            "Median": float(values.median()),
            "Minimum": float(values.min()),
            "Maximum": float(values.max()),
            "Std": float(values.std()),
            "High Trust (%)": float(
                (values >= 0.80).mean() * 100
            ),
            "Low Trust (%)": float(
                (values < 0.50).mean() * 100
            ),
        }

    # Calibration Scatter
    
    def calibration_scatter(
        self,
        df,
        confidence_col="Confidence",
        accuracy_col="Correct",
        title="Confidence Calibration",
    ):

        self.validate_columns(
            df,
            [confidence_col, accuracy_col],
        )

        fig = px.scatter(
            df,
            x=confidence_col,
            y=accuracy_col,
            trendline="ols",
        )

        return self.apply_layout(fig, title)

    # Trust Timeline by Model
    
    def trust_timeline(
        self,
        df,
        date_col="Date",
        trust_col="TrustScore",
        model_col="Model",
        title="Trust Timeline",
    ):

        self.validate_columns(
            df,
            [date_col, trust_col, model_col],
        )

        fig = px.line(
            df,
            x=date_col,
            y=trust_col,
            color=model_col,
            markers=True,
        )

        return self.apply_layout(fig, title)
    