import pandas as pd


class MarketRegimeDetector:
    @staticmethod
    def detect(df: pd.DataFrame):
        df = df.copy()
        regime = []

        for r in df["Return_5D"]:

            if pd.isna(r):
                regime.append("Unknown")

            elif r > 0.03:
                regime.append("Bull")

            elif r < -0.03:
                regime.append("Bear")

            else:
                regime.append("Sideways")

        df["MarketRegime"] = regime

        return df