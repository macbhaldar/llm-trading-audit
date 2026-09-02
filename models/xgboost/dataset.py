import pandas as pd

class DatasetBuilder:
    @staticmethod
    def build(df, target):
        X = df.drop(columns=[target])
        y = df[target]

        return X, y