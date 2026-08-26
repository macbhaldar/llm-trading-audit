import numpy as np


class PortfolioMetrics:

    @staticmethod
    def beta(
        portfolio,
        benchmark,
    ):

        cov = np.cov(
            portfolio,
            benchmark
        )[0, 1]

        var = np.var(
            benchmark
        )

        return cov / var

    @staticmethod
    def alpha(
        portfolio,
        benchmark,
        risk_free=0.02,
    ):

        beta = PortfolioMetrics.beta(
            portfolio,
            benchmark,
        )

        return (

            np.mean(portfolio)

            -

            risk_free

            -

            beta

            *

            (

                np.mean(benchmark)

                -

                risk_free

            )

        )
