class PromptBuilder:

    @staticmethod
    def build(
        ticker,
        close_price,
        sentiment,
        rsi,
        macd,
        regime,
    ):

        prompt = f"""
You are an experienced quantitative trader.

Analyze the following market information.

Ticker: {ticker}

Current Price: {close_price}

News Sentiment: {sentiment}

RSI: {rsi}

MACD: {macd}

Market Regime: {regime}

Return ONLY valid JSON.

Required format:

{{
"recommendation":"BUY/SELL/HOLD",
"confidence":0.0,
"target_price":0.0,
"stop_loss":0.0,
"holding_days":0,
"reasoning":"..."
}}
"""

        return prompt