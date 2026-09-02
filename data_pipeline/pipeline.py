from configs import config

from .data_loader import DataLoader
from .validator import DataValidator
from .preprocessing import Preprocessor
from .feature_engineering import FeatureEngineer
from .market_regime_detector import MarketRegimeDetector
from .merger import DatasetMerger


class DataPipeline:
    def __init__(self):
        self.loader = DataLoader(config)

    def run(self):
        prices = self.loader.load_market_prices()
        news = self.loader.load_news()
        macro = self.loader.load_macro()
        predictions = self.loader.load_llm_predictions()
        portfolio = self.loader.load_portfolio()
        DataValidator.validate(prices)
        prices = Preprocessor.clean(prices)
        prices = FeatureEngineer.create(prices)
        prices = MarketRegimeDetector.detect(prices)
        final_df = DatasetMerger.merge(
            prices,
            news,
            macro,
            predictions,
            portfolio
        )

        return final_df
    