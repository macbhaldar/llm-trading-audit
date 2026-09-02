from .regression_metrics import RegressionMetrics
from .classification_metrics import ClassificationMetrics
from .financial_metrics import FinancialMetrics


class EvaluationMetrics:
    @staticmethod
    def regression(
        y_true,
        y_pred,
    ):

        return RegressionMetrics.evaluate(
            y_true,
            y_pred,
        )

    @staticmethod
    def classification(
        y_true,
        y_pred,
    ):

        return ClassificationMetrics.evaluate(
            y_true,
            y_pred,
        )

    @staticmethod
    def finance(
        returns,
    ):

        return {

            "CumulativeReturn":
            FinancialMetrics.cumulative_return(
                returns
            ),

            "AnnualReturn":
            FinancialMetrics.annual_return(
                returns
            ),

            "Volatility":
            FinancialMetrics.volatility(
                returns
            ),

            "Sharpe":
            FinancialMetrics.sharpe_ratio(
                returns
            ),

            "Sortino":
            FinancialMetrics.sortino_ratio(
                returns
            ),

            "MaxDrawdown":
            FinancialMetrics.max_drawdown(
                returns
            ),
        }