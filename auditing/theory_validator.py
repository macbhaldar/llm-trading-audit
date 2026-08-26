class TheoryValidator:
    """
    Basic financial-theory consistency checks.
    """

    @staticmethod
    def validate(
        recommendation,
        rsi,
        sentiment,
    ):

        score = 1.0

        if recommendation == "BUY" and rsi > 80:
            score -= 0.30

        if recommendation == "SELL" and rsi < 20:
            score -= 0.30

        if recommendation == "BUY" and sentiment < 0:
            score -= 0.20

        if recommendation == "SELL" and sentiment > 0:
            score -= 0.20

        return max(score, 0)