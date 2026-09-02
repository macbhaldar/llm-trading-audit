import re

class LLMReasoningAnalyzer:
    POSITIVE = {
        "growth",
        "profit",
        "bullish",
        "momentum",
        "strong",
        "uptrend",
    }

    NEGATIVE = {
        "loss",
        "bearish",
        "weak",
        "risk",
        "downtrend",
        "decline",
    }

    @classmethod
    def analyze(
        cls,
        reasoning,
    ):

        text = reasoning.lower()

        words = set(
            re.findall(
                r"\b[a-z]+\b",
                text,
            )
        )

        positive = len(
            words & cls.POSITIVE
        )

        negative = len(
            words & cls.NEGATIVE
        )

        return {
            "PositiveTerms":
            positive,

            "NegativeTerms":
            negative,

            "NetSentiment":
            positive - negative,
        }