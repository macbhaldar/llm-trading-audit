import pandas as pd

class CounterfactualGenerator:
    """
    Simple rule-based counterfactual generator.
    """
    @staticmethod
    def generate(
        sample,
    ):

        cf = sample.copy()

        if "RSI" in cf:

            cf["RSI"] = 50

        if "Sentiment" in cf:

            cf["Sentiment"] *= -1

        if "MACD" in cf:

            cf["MACD"] *= -1

        return pd.Series(cf)
    