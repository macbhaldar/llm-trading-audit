from dataclasses import dataclass


@dataclass
class TradeRecommendation:

    ticker: str

    recommendation: str

    confidence: float

    target_price: float

    stop_loss: float

    holding_days: int

    reasoning: str