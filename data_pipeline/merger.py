import pandas as pd

class DatasetMerger:
    @staticmethod
    def merge(
        prices,
        news=None,
        macro=None,
        predictions=None,
        portfolio=None,
    ):

        df = prices.copy()

        if news is not None:
            df = df.merge(
                news,
                on=["Date", "Ticker"],
                how="left"
            )

        if macro is not None:
            df = df.merge(
                macro,
                on="Date",
                how="left"
            )

        if predictions is not None:
            df = df.merge(
                predictions,
                on=["Date", "Ticker"],
                how="left"
            )

        if portfolio is not None:
            df = df.merge(
                portfolio,
                on=["Date", "Ticker"],
                how="left"
            )

        return df