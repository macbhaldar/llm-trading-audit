import json
from .schemas import TradeRecommendation

class PredictionParser:
    @staticmethod
    def parse(text):
        data = json.loads(text)
        return TradeRecommendation(
            ticker=data.get("ticker", ""),
            recommendation=data["recommendation"],
            confidence=float(data["confidence"]),
            target_price=float(data["target_price"]),
            stop_loss=float(data["stop_loss"]),
            holding_days=int(data["holding_days"]),
            reasoning=data["reasoning"]
        )