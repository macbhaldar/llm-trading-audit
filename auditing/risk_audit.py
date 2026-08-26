import numpy as np


class RiskAudit:

    @staticmethod
    def maximum_drawdown(returns):

        cumulative = (
            1 + np.asarray(returns)
        ).cumprod()

        peak = np.maximum.accumulate(cumulative)

        drawdown = (
            cumulative - peak
        ) / peak

        return abs(drawdown.min())

    @staticmethod
    def value_at_risk(
        returns,
        alpha=0.05
    ):

        return np.percentile(
            returns,
            alpha * 100
        )

    @staticmethod
    def conditional_var(
        returns,
        alpha=0.05
    ):

        var = RiskAudit.value_at_risk(
            returns,
            alpha
        )

        return np.mean(
            np.asarray(returns)[
                np.asarray(returns) <= var
            ]
        )