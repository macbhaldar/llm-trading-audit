import pandas as pd


class HallucinationDetector:
    """
    Rule-based baseline hallucination detector.
    """

    KEYWORDS = [
        "guaranteed",
        "risk free",
        "always",
        "never",
        "100%",
        "certain",
        "impossible to lose"
    ]

    @classmethod
    def score(cls, reasoning):
        if pd.isna(reasoning):
            return 1.0

        text = reasoning.lower()

        hits = sum(
            word in text
            for word in cls.KEYWORDS
        )

        return min(hits / len(cls.KEYWORDS), 1.0)