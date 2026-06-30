from pathlib import Path


# Project Paths


ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"

RAW = DATA / "raw"

PROCESSED = DATA / "processed"

GT = DATA / "gt_tables"

REPORTS = ROOT / "reports"

MODELS = ROOT / "saved_models"


# Dataset Files

PRICE_DATA = RAW / "market_prices.csv"

NEWS_DATA = RAW / "market_news.csv"

MACRO_DATA = RAW / "macro_indicators.csv"

LLM_DATA = RAW / "llm_predictions.csv"

PORTFOLIO_DATA = RAW / "portfolio.csv"

GT_TABLE = GT / "gt_trade_table.csv"

AUDIT_RESULTS = GT / "audit_results.csv"

TRUST_TABLE = GT / "trust_scores.csv"


# Trading Parameters

LOOKBACK_WINDOW = 30

PREDICTION_HORIZON = 5

INITIAL_CAPITAL = 100000

RISK_FREE_RATE = 0.05

CONFIDENCE_THRESHOLD = 0.70


# Training

TEST_SIZE = 0.20

RANDOM_STATE = 42

N_ESTIMATORS = 300

MAX_DEPTH = 8

LEARNING_RATE = 0.05

# ======================================
# Audit Weights
# ======================================

TRUST_WEIGHTS = {
    "accuracy":0.30,
    "calibration":0.20,
    "hallucination":0.15,
    "risk":0.15,
    "theory":0.10,
    "explainability":0.10
}