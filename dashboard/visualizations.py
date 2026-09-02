"""
Dashboard Visualizations
Part 1 - Base Visualizer + Market Charts
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.config import (
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SUCCESS_COLOR,
    ERROR_COLOR,
    PLOTLY_TEMPLATE,
    DEFAULT_HEIGHT,
)

class DashboardVisualizer:
    """
    Reusable Plotly visualization library.
    Every method returns a Plotly Figure.
    """

    def __init__(self):
        self.template = PLOTLY_TEMPLATE
        self.primary = PRIMARY_COLOR
        self.secondary = SECONDARY_COLOR
        self.success = SUCCESS_COLOR
        self.error = ERROR_COLOR

    
    # Internal Helpers
    
    def _layout(
        self,
        fig,
        title: str,
        height: int = DEFAULT_HEIGHT,
    ):

        fig.update_layout(
            template=self.template,
            title=title,
            height=height,
            legend=dict(
                orientation="h",
                y=1.02,
                x=1,
                xanchor="right",
            ),

            margin=dict(
                l=40,
                r=20,
                t=60,
                b=40,
            ),
        )

        return fig

    
    # Line Chart

    def line_chart(
        self,
        df,
        x,
        y,
        color=None,
        title="Line Chart",
    ):

        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            markers=True,
        )
        return self._layout(fig, title)

    
    # Candlestick Chart
    
    def candlestick(
        self,
        df,
        title="Candlestick",
    ):
        
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=self.success,
                decreasing_line_color=self.error,
                name="Price",
            )
        )
        return self._layout(fig, title, 600)

    
    # Volume Chart
    
    def volume_chart(
        self,
        df,
        title="Trading Volume",
    ):

        fig = px.bar(
            df,
            x="Date",
            y="Volume",
        )
        return self._layout(fig, title)

    
    # Daily Returns
    
    def returns_chart(
        self,
        df,
        title="Daily Returns",
    ):

        data = df.copy()
        data["Return"] = data["Close"].pct_change()

        fig = px.line(
            data,
            x="Date",
            y="Return",
        )
        return self._layout(fig, title)

    
    # Histogram
    
    def return_distribution(
        self,
        df,
        title="Return Distribution",
    ):

        data = df.copy()
        data["Return"] = data["Close"].pct_change()

        fig = px.histogram(
            data,
            x="Return",
            nbins=40,
        )
        return self._layout(fig, title)

    
    # Moving Average

    def moving_average_chart(
        self,
        df,
        window=20,
        title="Moving Average",
    ):

        data = df.copy()
        data["MA"] = (
            data["Close"]
            .rolling(window)
            .mean()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                name="Close",
                line=dict(
                    color=self.primary
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["MA"],
                name=f"MA {window}",
                line=dict(
                    color=self.secondary
                ),
            )
        )

        return self._layout(fig, title)

    
    # Price + Volume

    def price_volume_chart(
        self,
        df,
        title="Price & Volume",
    ):

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.70, 0.30],
        )

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Close",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                name="Volume",
            ),
            row=2,
            col=1,
        )
        return self._layout(fig, title, 650)

    
    # Market Regime
    
    def market_regime_chart(
        self,
        df,
        regime_column="MarketRegime",
        title="Market Regime",
    ):

        fig = px.scatter(
            df,
            x="Date",
            y="Close",
            color=regime_column,
        )

        return self._layout(fig, title)

    
    # Correlation Heatmap
    
    def correlation_heatmap(
        self,
        df,
        title="Correlation Matrix",
    ):

        corr = (
            df
            .select_dtypes("number")
            .corr()
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
        )

        return self._layout(fig, title)

    
    # OHLC Chart

    def ohlc_chart(
        self,
        df,
        title="OHLC",
    ):

        fig = go.Figure(
            go.Ohlc(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        )
        return self._layout(fig, title)

    
    # Rolling Volatility
    
    def rolling_volatility(
        self,
        df,
        window=20,
        title="Rolling Volatility",
    ):
        data = df.copy()
        data["Return"] = data["Close"].pct_change()
        data["Volatility"] = (
            data["Return"]
            .rolling(window)
            .std()
        )

        fig = px.line(
            data,
            x="Date",
            y="Volatility",
        )
        return self._layout(fig, title)


    # Rolling Mean
    def rolling_mean(
        self,
        df,
        window=20,
        title="Rolling Mean",
    ):
        data = df.copy()

        data["RollingMean"] = (
            data["Close"]
            .rolling(window)
            .mean()
        )
        fig = px.line(
            data,
            x="Date",
            y="RollingMean",
        )
        return self._layout(fig, title)
    