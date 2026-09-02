import numpy as np

class Preprocessor:
    @staticmethod
    def clean(df):
        df = df.copy()
        df = df.drop_duplicates()

        numeric = df.select_dtypes(include=np.number).columns
        categorical = df.select_dtypes(exclude=np.number).columns

        for col in numeric:
            df[col] = df[col].fillna(df[col].median())

        for col in categorical:
            df[col] = df[col].fillna("Unknown")

        return df