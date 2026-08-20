from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Load every dataset used by the project.
    """

    def __init__(self, config):
        self.config = config

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        return pd.read_csv(path)

    def load_market_prices(self):
        return self._load(self.config.PRICE_DATA)

    def load_news(self):
        return self._load(self.config.NEWS_DATA)

    def load_macro(self):
        return self._load(self.config.MACRO_DATA)

    def load_llm_predictions(self):
        return self._load(self.config.LLM_DATA)

    def load_portfolio(self):
        return self._load(self.config.PORTFOLIO_DATA)