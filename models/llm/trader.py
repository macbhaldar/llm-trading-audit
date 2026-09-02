import json
from .prompt_builder import PromptBuilder
from .schemas import TradeRecommendation

class DummyLLM:
    """
    Replace this class with:
    OpenAI
    Ollama
    HuggingFace
    vLLM
    Azure OpenAI
    etc.
    """

    def generate(self, prompt):
        prediction = {
            "ticker": "AAPL",
            "recommendation": "BUY",
            "confidence": 0.89,
            "target_price": 214.75,
            "stop_loss": 185.00,
            "holding_days": 10,
            "reasoning":
            "Positive sentiment with bullish momentum and healthy RSI."
        }
        return json.dumps(prediction)

class LLMTrader:
    def __init__(self, llm=None):
        self.llm = llm or DummyLLM()

    def predict(
        self,
        ticker,
        close_price,
        sentiment,
        rsi,
        macd,
        regime,
    ):

        prompt = PromptBuilder.build(
            ticker,
            close_price,
            sentiment,
            rsi,
            macd,
            regime
        )

        response = self.llm.generate(prompt)

        prediction = json.loads(response)

        return TradeRecommendation(
            ticker=ticker,
            recommendation=prediction["recommendation"],
            confidence=prediction["confidence"],
            target_price=prediction["target_price"],
            stop_loss=prediction["stop_loss"],
            holding_days=prediction["holding_days"],
            reasoning=prediction["reasoning"]
        )