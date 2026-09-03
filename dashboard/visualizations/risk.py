"""
Portfolio Risk Visualizations

"""

from typing import Optional

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from .base import BaseVisualizer


class RiskVisualizer(BaseVisualizer):

    # Value at Risk (VaR)
    
    def var_chart(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        title: str = "Value at Risk (VaR)"
    ):

        returns = pd.Series(returns).dropna()

        var = np.percentile(
            returns,
            (1 - confidence) * 100
        )

        fig = px.histogram(
            x=returns,
            nbins=50,
            labels={"x": "Returns"},
        )

        fig.add_vline(
            x=var,
            line_color="red",
            line_width=3,
            annotation_text=f"VaR ({confidence:.0%})",
        )

        return self.apply_layout(fig, title)

    # Conditional VaR
    
    def cvar_chart(
        self,
        returns,
        confidence=0.95,
        title="Conditional VaR (CVaR)"
    ):

        returns = pd.Series(returns).dropna()

        var = np.percentile(
            returns,
            (1 - confidence) * 100
        )

        cvar = returns[
            returns <= var
        ].mean()

        fig = px.histogram(
            x=returns,
            nbins=50,
        )

        fig.add_vline(
            x=var,
            line_color="red",
            annotation_text="VaR",
        )

        fig.add_vline(
            x=cvar,
            line_color="purple",
            annotation_text="CVaR",
        )

        return self.apply_layout(fig, title)

    # Drawdown
    
    def drawdown(
        self,
        equity,
        dates=None,
        title="Portfolio Drawdown"
    ):

        equity = pd.Series(equity)

        running_max = equity.cummax()

        drawdown = (
            equity - running_max
        ) / running_max

        fig = px.area(
            x=dates if dates is not None else equity.index,
            y=drawdown,
            labels={
                "x": "Date",
                "y": "Drawdown",
            },
        )

        return self.apply_layout(fig, title)

    # Rolling Volatility
    
    def rolling_volatility(
        self,
        returns,
        window=30,
        title="Rolling Volatility"
    ):

        returns = pd.Series(returns)

        vol = (
            returns
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

        fig = px.line(
            y=vol,
            x=vol.index,
        )

        return self.apply_layout(fig, title)

    # Risk Contribution
    
    def risk_contribution(
        self,
        contribution_df,
        asset_col="Asset",
        value_col="Contribution",
        title="Risk Contribution"
    ):

        self.validate_columns(
            contribution_df,
            [
                asset_col,
                value_col,
            ],
        )

        fig = px.bar(
            contribution_df,
            x=asset_col,
            y=value_col,
            color=value_col,
        )

        return self.apply_layout(fig, title)

    # Exposure
    
    def exposure_chart(
        self,
        exposure_df,
        asset_col="Asset",
        exposure_col="Exposure",
        title="Portfolio Exposure"
    ):

        self.validate_columns(
            exposure_df,
            [
                asset_col,
                exposure_col,
            ],
        )

        fig = px.pie(
            exposure_df,
            names=asset_col,
            values=exposure_col,
        )

        return self.apply_layout(fig, title)

        # Sector Exposure
    
    def sector_exposure(
        self,
        df,
        sector_col="Sector",
        weight_col="Weight",
        title="Sector Allocation"
    ):

        self.validate_columns(
            df,
            [
                sector_col,
                weight_col,
            ],
        )

        fig = px.treemap(
            df,
            path=[sector_col],
            values=weight_col,
        )

        return self.apply_layout(fig, title)

    # Risk Heatmap
    
    def correlation_heatmap(
        self,
        df,
        title="Risk Correlation Matrix"
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

        return self.apply_layout(fig, title)

    # Leverage
    
    def leverage_chart(
        self,
        df,
        date_col="Date",
        leverage_col="Leverage",
        title="Portfolio Leverage"
    ):

        self.validate_columns(
            df,
            [
                date_col,
                leverage_col,
            ],
        )

        fig = px.line(
            df,
            x=date_col,
            y=leverage_col,
        )

        return self.apply_layout(fig, title)

    # Stress Testing
    
    def stress_test(
        self,
        df,
        scenario_col="Scenario",
        pnl_col="PnL",
        title="Stress Test Results"
    ):

        self.validate_columns(
            df,
            [
                scenario_col,
                pnl_col,
            ],
        )

        fig = px.bar(
            df,
            x=scenario_col,
            y=pnl_col,
            color=pnl_col,
        )

        return self.apply_layout(fig, title)

    # Sharpe vs Volatility
    
    def sharpe_vs_volatility(
        self,
        df,
        sharpe_col="Sharpe",
        volatility_col="Volatility",
        label_col: Optional[str] = "Model",
        title="Sharpe Ratio vs Volatility"
    ):

        self.validate_columns(
            df,
            [
                sharpe_col,
                volatility_col,
            ],
        )

        fig = px.scatter(
            df,
            x=volatility_col,
            y=sharpe_col,
            text=label_col,
            size=sharpe_col,
        )
        fig.update_traces(
            textposition="top center"
        )

        return self.apply_layout(fig, title)

    # Risk Summary
    
    def risk_summary(
        self,
        returns
    ):

        returns = pd.Series(returns).dropna()

        var95 = np.percentile(
            returns,
            5
        )

        cvar95 = returns[
            returns <= var95
        ].mean()

        return {
            "Mean Return": float(returns.mean()),
            "Volatility": float(returns.std()),
            "Minimum": float(returns.min()),
            "Maximum": float(returns.max()),
            "VaR95": float(var95),
            "CVaR95": float(cvar95),
            "Downside Risk": float(
                returns[
                    returns < 0
                ].std()
            ),
        }
    