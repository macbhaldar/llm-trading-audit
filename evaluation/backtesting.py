import pandas as pd


class Backtester:

    def run(
        self,
        prices,
        signals,
    ):

        df = prices.copy()

        df["Signal"] = signals

        df["MarketReturn"] = (
            df["Close"].pct_change()
        )

        df["StrategyReturn"] = (

            df["Signal"].shift(1)

            *

            df["MarketReturn"]

        )

        return df
