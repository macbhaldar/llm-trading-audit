import numpy as np


class FinancialMetrics:

    @staticmethod
    def cumulative_return(returns):

        returns = np.asarray(returns)

        return (1 + returns).prod() - 1

    @staticmethod
    def annual_return(
        returns,
        periods=252,
    ):

        returns = np.asarray(returns)

        return (
            (1 + returns).prod()
            ** (periods / len(returns))
        ) - 1

    @staticmethod
    def volatility(
        returns,
        periods=252,
    ):

        returns = np.asarray(returns)

        return (
            np.std(returns)
            * np.sqrt(periods)
        )

    @staticmethod
    def sharpe_ratio(
        returns,
        risk_free=0.02,
        periods=252,
    ):

        annual = FinancialMetrics.annual_return(
            returns,
            periods,
        )

        vol = FinancialMetrics.volatility(
            returns,
            periods,
        )

        return (annual - risk_free) / vol

    @staticmethod
    def sortino_ratio(
        returns,
        risk_free=0.02,
        periods=252,
    ):

        returns = np.asarray(returns)

        downside = returns[returns < 0]

        if len(downside) == 0:
            return np.inf

        downside_std = (
            np.std(downside)
            * np.sqrt(periods)
        )

        annual = FinancialMetrics.annual_return(
            returns,
            periods,
        )

        return (
            annual - risk_free
        ) / downside_std

    @staticmethod
    def max_drawdown(returns):

        wealth = (
            1 + np.asarray(returns)
        ).cumprod()

        peak = np.maximum.accumulate(wealth)

        dd = (wealth - peak) / peak

        return abs(dd.min())