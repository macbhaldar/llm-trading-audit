import pandas as pd


class FeatureEngineer:

    @staticmethod
    def create(df: pd.DataFrame):

        df = df.copy()

        df["Return_1D"] = df["Close"].pct_change()

        df["Return_5D"] = df["Close"].pct_change(5)

        df["MA20"] = (
            df["Close"]
            .rolling(20)
            .mean()
        )

        df["MA50"] = (
            df["Close"]
            .rolling(50)
            .mean()
        )

        df["Volatility20"] = (
            df["Return_1D"]
            .rolling(20)
            .std()
        )

        df["Momentum10"] = (
            df["Close"]
            -
            df["Close"].shift(10)
        )

        return df