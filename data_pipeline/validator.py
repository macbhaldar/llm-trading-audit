import pandas as pd

class DataValidator:
    @staticmethod
    def validate(df: pd.DataFrame):
        if df.empty:
            raise ValueError("Dataset is empty.")
        duplicated = df.duplicated().sum()
        missing = df.isna().sum()

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicates": duplicated,
            "missing": missing.to_dict()
        }

    @staticmethod
    def require_columns(df, columns):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )
