from pathlib import Path


# Project Directories

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

GT_DATA_DIR = DATA_DIR / "gt_tables"

MODEL_DIR = ROOT_DIR / "saved_models"

REPORT_DIR = ROOT_DIR / "reports"

ASSET_DIR = ROOT_DIR / "dashboard" / "assets"

LOG_DIR = ROOT_DIR / "logs"


# Dashboard

APP_NAME = "Auditing LLM Trading"

APP_VERSION = "1.0.0"

PAGE_TITLE = "Auditing LLM Trading Dashboard"

PAGE_ICON = "📈"

LAYOUT = "wide"

SIDEBAR_STATE = "expanded"


# Theme

PRIMARY_COLOR = "#2563EB"

SECONDARY_COLOR = "#14B8A6"

SUCCESS_COLOR = "#10B981"

WARNING_COLOR = "#F59E0B"

ERROR_COLOR = "#EF4444"

BACKGROUND_COLOR = "#F8FAFC"

TEXT_COLOR = "#111827"

GRID_COLOR = "#E5E7EB"


# Plot Sizes

DEFAULT_HEIGHT = 450

LARGE_HEIGHT = 700

SMALL_HEIGHT = 250

DEFAULT_WIDTH = None


# Plotly Theme

PLOTLY_TEMPLATE = "plotly_white"


# Default Models

AVAILABLE_MODELS = [

    "LLM",

    "Transformer",

    "LSTM",

    "XGBoost"

]


# Market Regimes

MARKET_REGIMES = [

    "Bull",

    "Bear",

    "Sideways"

]


# Trading Signals

SIGNALS = [

    "BUY",

    "SELL",

    "HOLD"

]


# Risk Settings

RISK_FREE_RATE = 0.02

TRADING_DAYS = 252

CONFIDENCE_LEVEL = 0.95


# Dashboard Refresh

AUTO_REFRESH = False

REFRESH_SECONDS = 30


# Table Configuration

TABLE_PAGE_SIZE = 25

TABLE_HEIGHT = 650


# Default Charts

CHART_OPTIONS = [

    "Price History",

    "Returns",

    "Drawdown",

    "Trust Score",

    "Calibration",

    "Hallucination",

    "Confusion Matrix",

    "Feature Importance",

    "Attention Heatmap",

    "SHAP Summary"

]


# Export Options

EXPORT_FORMATS = [

    "CSV",

    "Excel",

    "PDF",

    "HTML",

    "JSON"

]


# Dashboard Pages

PAGES = [

    "Overview",

    "Ground Truth",

    "Predictions",

    "Trust Score",

    "Calibration",

    "Hallucination",

    "Explainability",

    "Backtesting",

    "Performance",

    "Leaderboard",

    "Settings"

]


# Required Data Files

REQUIRED_FILES = {

    "prices":
        RAW_DATA_DIR / "market_prices.csv",

    "predictions":
        RAW_DATA_DIR / "llm_predictions.csv",

    "gt":
        GT_DATA_DIR / "gt_trade_table.csv",

    "trust":
        GT_DATA_DIR / "trust_scores.csv",

    "backtest":
        PROCESSED_DATA_DIR / "backtest_results.csv",

    "metrics":
        PROCESSED_DATA_DIR / "evaluation_metrics.csv"

}


# Column Names

DATE_COLUMN = "Date"

TICKER_COLUMN = "Ticker"

PRICE_COLUMN = "Close"

RETURN_COLUMN = "Return"

PREDICTION_COLUMN = "Recommendation"

CONFIDENCE_COLUMN = "Confidence"

TRUST_COLUMN = "TrustScore"


# Logging

LOG_LEVEL = "INFO"

LOG_FILE = LOG_DIR / "dashboard.log"