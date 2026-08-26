import numpy as np
import pandas as pd


class GTGenerator:
    """
    Generate Ground Truth (GT) table by comparing predictions
    with realized market returns.
    """

    def __init__(self, horizon=5):

        self.horizon = horizon

    def generate(self, predictions, prices):

        df = predictions.copy()

        price = prices.copy()

        price = price.sort_values(["Ticker", "Date"])

        future_close = (
            price.groupby("Ticker")["Close"]
            .shift(-self.horizon)
        )

        price["FutureClose"] = future_close

        merged = df.merge(
            price[
                [
                    "Date",
                    "Ticker",
                    "Close",
                    "FutureClose"
                ]
            ],
            on=["Date", "Ticker"],
            how="left"
        )

        merged["ActualReturn"] = (
            merged["FutureClose"] -
            merged["Close"]
        ) / merged["Close"]

        def evaluate(row):

            pred = row["Recommendation"]

            ret = row["ActualReturn"]

            if pd.isna(ret):
                return np.nan

            if pred == "BUY":
                return ret > 0

            if pred == "SELL":
                return ret < 0

            return abs(ret) < 0.01

        merged["Correct"] = merged.apply(
            evaluate,
            axis=1
        )

        return merged