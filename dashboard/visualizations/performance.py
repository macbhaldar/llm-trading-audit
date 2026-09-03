"""
Performance Visualizations : Reusable charts for portfolio performance, strategy evaluation, and backtesting.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .base import BaseVisualizer


class PerformanceVisualizer(BaseVisualizer):

    # Equity Curve
    
    def equity_curve(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
        equity_col: str = "PortfolioValue",
        title: str = "Portfolio Equity Curve",
    ) -> go.Figure:

        self.validate_columns(df, [date_col, equity_col])
        fig = px.line(
            df,
            x=date_col,
            y=equity_col,
        )

        fig.update_traces(
            line=dict(width=3)
        )

        return self.apply_layout(fig, title)

    # Benchmark Comparison
    
    def benchmark_comparison(
        self,
        df: pd.DataFrame,
        portfolio_col="PortfolioValue",
        benchmark_col="Benchmark",
        date_col="Date",
        title="Portfolio vs Benchmark",
    ):

        self.validate_columns(
            df,
            [
                date_col,
                portfolio_col,
                benchmark_col,
            ],
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[portfolio_col],
                mode="lines",
                name="Portfolio",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[benchmark_col],
                mode="lines",
                name="Benchmark",
            )
        )

        return self.apply_layout(fig, title)

    # Cumulative Returns
    
    def cumulative_returns(
        self,
        df,
        return_col="Return",
        date_col="Date",
        title="Cumulative Returns",
    ):

        self.validate_columns(
            df,
            [date_col, return_col],
        )

        data = df.copy()
        data["Cumulative"] = (
            1 + data[return_col]
        ).cumprod() - 1

        fig = px.line(
            data,
            x=date_col,
            y="Cumulative",
        )

        return self.apply_layout(fig, title)

    # Drawdown
    
    def drawdown_chart(
        self,
        df,
        equity_col="PortfolioValue",
        date_col="Date",
        title="Drawdown",
    ):

        self.validate_columns(
            df,
            [date_col, equity_col],
        )

        data = df.copy()
        running_max = data[equity_col].cummax()

        data["Drawdown"] = (
            data[equity_col] - running_max
        ) / running_max

        fig = px.area(
            data,
            x=date_col,
            y="Drawdown",
        )

        return self.apply_layout(fig, title)

    # Rolling Sharpe Ratio
    
    def rolling_sharpe(
        self,
        df,
        return_col="Return",
        date_col="Date",
        window=30,
        risk_free_rate=0.0,
        title="Rolling Sharpe Ratio",
    ):

        self.validate_columns(
            df,
            [date_col, return_col],
        )

        data = df.copy()
        excess = (
            data[return_col] - risk_free_rate / 252
        )

        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()

        data["Sharpe"] = (
            np.sqrt(252)
            * rolling_mean
            / rolling_std
        )

        fig = px.line(
            data,
            x=date_col,
            y="Sharpe",
        )

        return self.apply_layout(fig, title)

    # Rolling Volatility
    
    def rolling_volatility(
        self,
        df,
        return_col="Return",
        date_col="Date",
        window=30,
        title="Rolling Volatility",
    ):

        self.validate_columns(
            df,
            [date_col, return_col],
        )

        data = df.copy()
        data["Volatility"] = (
            data[return_col]
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

        fig = px.line(
            data,
            x=date_col,
            y="Volatility",
        )

        return self.apply_layout(fig, title)

    # Monthly Returns Heatmap
    
    def monthly_returns_heatmap(
        self,
        df,
        date_col="Date",
        return_col="Return",
        title="Monthly Returns",
    ):

        self.validate_columns(
            df,
            [date_col, return_col],
        )

        data = df.copy()
        data[date_col] = pd.to_datetime(
            data[date_col]
        )

        data["Year"] = data[date_col].dt.year
        data["Month"] = data[date_col].dt.month_name()

        pivot = data.pivot_table(
            index="Year",
            columns="Month",
            values=return_col,
            aggfunc="mean",
        )

        fig = px.imshow(
            pivot,
            text_auto=".2%",
            aspect="auto",
        )

        return self.apply_layout(fig, title)

    # Trade Distribution
    
    def trade_distribution(
        self,
        df,
        pnl_col="PnL",
        title="Trade Distribution",
    ):
        self.validate_columns(
            df,
            [pnl_col],
        )
        fig = px.histogram(
            df,
            x=pnl_col,
            nbins=40,
        )

        return self.apply_layout(fig, title)

    # Win/Loss Pie
    
    def win_loss_chart(
        self,
        df,
        pnl_col="PnL",
        title="Winning vs Losing Trades",
    ):
        self.validate_columns(
            df,
            [pnl_col],
        )
        wins = (df[pnl_col] > 0).sum()
        losses = (df[pnl_col] <= 0).sum()

        fig = px.pie(
            names=["Win", "Loss"],
            values=[wins, losses],
        )

        return self.apply_layout(fig, title)

    # Trade Markers
    
    def trade_markers(
        self,
        df,
        date_col="Date",
        price_col="Close",
        signal_col="Signal",
        title="Trade Signals",
    ):

        self.validate_columns(
            df,
            [
                date_col,
                price_col,
                signal_col,
            ],
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[price_col],
                mode="lines",
                name="Price",
            )
        )

        buy = df[df[signal_col] == "BUY"]

        sell = df[df[signal_col] == "SELL"]

        fig.add_trace(
            go.Scatter(
                x=buy[date_col],
                y=buy[price_col],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                ),
                name="BUY",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sell[date_col],
                y=sell[price_col],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                ),
                name="SELL",
            )
        )

        return self.apply_layout(fig, title)
    