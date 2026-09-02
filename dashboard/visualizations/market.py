"""
Market Visualizations : Charts related to financial markets.
"""

from turtle import title

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base import BaseVisualizer


class MarketVisualizer(BaseVisualizer):
    """
    Market Charts
    """

    # Line Chart
    def line_chart(
        self,
        df: pd.DataFrame,
        x="Date",
        y="Close",
        color=None,
        title="Price History",
    ):

        self.validate_columns(df, [x, y])

        fig = px.line(
            df,
            x=x,
            y=y,
            color=color,
            markers=True,
        )

        return self.apply_layout(
            fig,
            title,
        )

    # Candlestick Chart

    def candlestick(
        self,
        df,
        title="Candlestick",
    ):

        self.validate_columns(
            df,
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
            ],
        )

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    increasing_line_color=self.success,
                    decreasing_line_color=self.error,
                )
            ]
        )

        return self.apply_layout(
            fig,
            title,
            650,
        )

    # OHLC Chart

    def ohlc_chart(
        self,
        df,
        title="OHLC",
    ):

        self.validate_columns(
            df,
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
            ],
        )

        fig = go.Figure(
            go.Ohlc(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        )
        return self.apply_layout(
            fig,
            title,
        )

    # Volume Chart

    def volume_chart(
        self,
        df,
        title="Volume",
    ):

        self.validate_columns(
            df,
            [
                "Date",
                "Volume",
            ],
        )

        fig = px.bar(
            df,
            x="Date",
            y="Volume",
        )

        return self.apply_layout(
            fig,
            title,
        )

    # Daily Returns Chart

    def returns_chart(
        self,
        df,
        title="Daily Returns",
    ):

        self.validate_columns(
            df,
            [
                "Date",
                "Close",
            ],
        )

        data = df.copy()

        data["Return"] = (
            data["Close"]
            .pct_change()
        )

        fig = px.line(
            data,
            x="Date",
            y="Return",
        )

        return self.apply_layout(
            fig,
            title,
        )

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


    # Moving Average Chart

    def moving_average(
        self,
        df,
        window=20,
        title="Moving Average",
    ):

        self.validate_columns(
            df,
            [
                "Date",
                "Close",
            ],
        )

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
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["MA"],
                name=f"MA {window}",
                mode="lines",
            )
        )

        return self.apply_layout(
            fig,
            title,
        )

    # Price & Volume Chart

    def price_volume(
        self,
        df,
        title="Price & Volume",
    ):

        self.validate_columns(
            df,
            [
                "Date",
                "Close",
                "Volume",
            ],
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.70, 0.30],
            vertical_spacing=0.03,
        )

        fig.add_trace(

            go.Scatter(
                x=df["Date"],
                y=df["Close"],
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

        return self.apply_layout(
            fig,
            title,
            700,
        )

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

        numeric = df.select_dtypes(
            include="number"
        )

        corr = numeric.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
        )

        return self.apply_layout(
            fig,
            title,
        )

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

    
    # Rolling Volatility Chart

    def rolling_volatility(
        self,
        df,
        window=20,
        title="Rolling Volatility",
    ):

        self.validate_columns(
            df,
            [
                "Date",
                "Close",
            ],
        )

        data = df.copy()

        data["Return"] = (
            data["Close"]
            .pct_change()
        )

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

        return self.apply_layout(
            fig,
            title,
        )

    # Rolling Mean Chart

    def rolling_mean(
        self,
        df,
        window=20,
        title="Rolling Mean",
    ):

        self.validate_columns(
            df,

            [
                "Date",
                "Close",
            ],
        )

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

        return self.apply_layout(
            fig,
            title,
        )