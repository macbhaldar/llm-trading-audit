from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Project Root

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import Project Modules

from configs import config

from data_pipeline import DataPipeline
from data_pipeline.data_loader import DataLoader

from auditing import AuditPipeline, GTGenerator
from auditing.calibration import Calibration

from evaluation import (
    EvaluationMetrics,
    Backtester,
    ModelComparison,
)

from explainability import ExplainabilityPipeline


# Logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "training.log",
            mode="w",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("main")


# Globals

DEFAULT_SEED = config.RANDOM_STATE

# Artifacts shared between steps (train -> explain)
SESSION = {
    "xgb_model": None,
    "xgb_features": None,
    "xgb_train_data": None,
}

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Volume",
    "VWAP",
    "Returns",
    "RSI",
    "MACD",
]


# Helpers

def banner(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def set_random_seed(seed: int = DEFAULT_SEED) -> None:
    """Make experiments reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    logger.info("Random seed set to %d", seed)


def execution_timer(func):
    """Log the wall-clock duration of a pipeline step."""

    def wrapper(*args, **kwargs):
        start = time.time()

        result = func(*args, **kwargs)

        logger.info(
            "%s completed in %.2f seconds",
            func.__name__,
            time.time() - start,
        )

        return result

    return wrapper


# Step 1 : Data Pipeline

@execution_timer
def run_pipeline() -> pd.DataFrame:
    banner("STEP 1/5 : Data Pipeline")

    pipeline = DataPipeline()

    df = pipeline.run()

    out_path = config.PROCESSED / "merged_dataset.csv"

    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)

    logger.info(
        "Merged dataset saved to %s (%d rows, %d columns)",
        out_path,
        len(df),
        df.shape[1],
    )

    print(df.head())

    print()
    print("Data pipeline finished.")

    return df


# Step 2 : Train Models

def _load_prices() -> pd.DataFrame:
    loader = DataLoader(config)

    return loader.load_market_prices()


def _build_supervised(prices: pd.DataFrame):
    """
    Per-ticker supervised frame: predict the next-day
    return from today's market features (no look-ahead).
    """

    frames = []

    for _, grp in prices.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()

        grp["Target"] = grp["Close"].pct_change().shift(-1)

        frames.append(grp)

    data = pd.concat(frames).dropna(
        subset=FEATURE_COLUMNS + ["Target"]
    )

    X = data[FEATURE_COLUMNS]
    y = data["Target"]

    return X, y


def _build_sequences(
    prices: pd.DataFrame,
    seq_len: int,
):
    """
    Sliding windows for LSTM / Transformer:
    X[i] = features of days i .. i+seq_len-1,
    y[i] = next-day return after the window.
    """

    X_seq, y_seq = [], []

    for _, grp in prices.groupby("Ticker"):
        grp = grp.sort_values("Date")

        feats = grp[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        feats = np.nan_to_num(feats)

        target = grp["Close"].pct_change().to_numpy()

        for i in range(len(feats) - seq_len):
            nxt = target[i + seq_len]

            if np.isnan(nxt):
                continue

            X_seq.append(feats[i : i + seq_len])
            y_seq.append(nxt)

    return (
        np.stack(X_seq),
        np.asarray(y_seq, dtype=np.float32),
    )


@execution_timer
def train_models(epochs: int = 3) -> dict:
    banner("STEP 2/5 : Model Training")

    prices = _load_prices()

    X, y = _build_supervised(prices)

    logger.info(
        "Supervised dataset: %d samples, %d features",
        len(X),
        len(FEATURE_COLUMNS),
    )

    results = {}

    # XGBoost baseline
    print("Training XGBoost...")

    try:
        from models.xgboost.config import XGBoostConfig
        from models.xgboost.trainer import XGBoostTrainer

        trainer = XGBoostTrainer(XGBoostConfig())

        model, X_tr, X_te, y_tr, y_te = trainer.train(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )

        metrics = EvaluationMetrics.regression(
            y_te,
            model.predict(X_te),
        )

        results["XGBoost"] = metrics

        SESSION["xgb_model"] = model
        SESSION["xgb_features"] = FEATURE_COLUMNS
        SESSION["xgb_train_data"] = X_tr

        config.MODELS.mkdir(parents=True, exist_ok=True)

        model_path = config.MODELS / "xgboost_model.json"

        model.save(str(model_path))

        logger.info(
            "XGBoost trained. Metrics: %s | saved to %s",
            {k: round(v, 6) for k, v in metrics.items()},
            model_path,
        )

        print(
            "XGBoost OK -> "
            + ", ".join(
                f"{k}={v:.6f}" for k, v in metrics.items()
            )
        )

    except Exception as exc:
        logger.error("XGBoost training skipped: %s", exc)

    # Deep learning baselines

    try:
        import torch

        from models.lstm.config import LSTMConfig
        from models.lstm.model import LSTMPricePredictor
        from models.lstm.trainer import LSTMTrainer
        from models.lstm.dataset import MarketDataset
        from models.lstm.predictor import LSTMPredictor

        from models.transformer.config import (
            TransformerConfig,
        )
        from models.transformer.model import (
            TransformerPricePredictor,
        )
        from models.transformer.trainer import (
            TransformerTrainer,
        )
        from models.transformer.predictor import (
            TransformerPredictor,
        )

        seq_len = config.LOOKBACK_WINDOW

        X_seq, y_seq = _build_sequences(
            prices,
            seq_len,
        )

        split = int(len(X_seq) * (1 - config.TEST_SIZE))

        X_tr_seq, X_te_seq = X_seq[:split], X_seq[split:]
        y_tr_seq, y_te_seq = y_seq[:split], y_seq[split:]

        train_ds = MarketDataset(X_tr_seq, y_tr_seq)

        input_size = X_seq.shape[2]

        # LSTM

        print("Training LSTM...")

        lstm_cfg = LSTMConfig(
            input_size=input_size,
            epochs=epochs,
            device="cpu",
        )

        lstm_model = LSTMPricePredictor(lstm_cfg)

        LSTMTrainer(lstm_model, lstm_cfg, train_ds).train()

        lstm_pred = LSTMPredictor(lstm_model).predict(
            X_te_seq
        )

        results["LSTM"] = {
            "RMSE": float(
                np.sqrt(
                    np.mean((lstm_pred - y_te_seq) ** 2)
                )
            )
        }

        print(
            "LSTM OK -> RMSE="
            f"{results['LSTM']['RMSE']:.6f}"
        )

        # Transformer

        print("Training Transformer...")

        tf_cfg = TransformerConfig(
            input_size=input_size,
            epochs=epochs,
            device="cpu",
        )

        tf_model = TransformerPricePredictor(tf_cfg)

        TransformerTrainer(
            tf_model,
            tf_cfg,
            train_ds,
        ).train()

        tf_pred = TransformerPredictor(tf_model).predict(
            X_te_seq
        )

        results["Transformer"] = {
            "RMSE": float(
                np.sqrt(
                    np.mean(
                        (tf_pred.ravel() - y_te_seq) ** 2
                    )
                )
            )
        }

        print(
            "Transformer OK -> RMSE="
            f"{results['Transformer']['RMSE']:.6f}"
        )

    except Exception as exc:
        logger.error(
            "Deep learning training skipped: %s", exc
        )

    # LLM trader (no training: prompt-driven)

    print("Running LLM trader demo...")

    try:
        from models.llm.trader import LLMTrader

        row = prices.iloc[0]

        recommendation = LLMTrader().predict(
            ticker=row["Ticker"],
            close_price=float(row["Close"]),
            sentiment=0.6,
            rsi=float(row["RSI"]),
            macd=float(row["MACD"]),
            regime="Sideways",
        )

        logger.info(
            "LLM demo recommendation: %s (%s, confidence %.2f)",
            recommendation.recommendation,
            recommendation.ticker,
            recommendation.confidence,
        )

        print(
            f"LLM demo -> {recommendation.ticker} "
            f"{recommendation.recommendation} "
            f"(confidence {recommendation.confidence:.2f})"
        )

        SESSION["llm_reasoning"] = recommendation.reasoning

    except Exception as exc:
        logger.error("LLM trader demo skipped: %s", exc)

    # Comparison

    if results:
        print()
        print(ModelComparison.compare(results))

    print()
    print("Training complete.")

    return results


# Step 3 : Auditing Engine (GT Table + Trust Scores)

@execution_timer
def run_audit() -> pd.DataFrame:
    banner("STEP 3/5 : Auditing Engine")

    loader = DataLoader(config)

    predictions = loader.load_llm_predictions()
    prices = loader.load_market_prices()
    news = loader.load_news()

    # Ground Truth table

    gt = GTGenerator(
        horizon=config.PREDICTION_HORIZON
    ).generate(predictions, prices)

    config.GT.mkdir(parents=True, exist_ok=True)

    gt.to_csv(config.GT_TABLE, index=False)

    logger.info(
        "GT table saved to %s (%d rows)",
        config.GT_TABLE,
        len(gt),
    )

    # Lookups for the audit modules

    rsi_lookup = prices.set_index(["Date", "Ticker"])[
        "RSI"
    ].to_dict()

    sentiment_lookup = (
        news.groupby(["Date", "Ticker"])["Sentiment"]
        .mean()
        .to_dict()
    )

    # Audit every prediction

    auditor = AuditPipeline()

    audit_rows = []

    skipped = 0

    for _, row in gt.iterrows():

        correct = row.get("Correct")

        if pd.isna(correct):
            skipped += 1
            continue

        key = (row["Date"], row["Ticker"])

        prediction = {
            "Recommendation": row["Recommendation"],
            "Confidence": float(row["Confidence"]),
            "Reasoning": row.get("Reasoning", ""),
        }

        result = auditor.audit(
            prediction,
            actual_correct=float(correct),
            rsi=rsi_lookup.get(key, 50.0),
            sentiment=sentiment_lookup.get(key, 0.0),
        )

        audit_rows.append(
            {
                "AuditID": len(audit_rows) + 1,
                "PredictionID": row["PredictionID"],
                "HallucinationScore": result[
                    "HallucinationScore"
                ],
                "CalibrationError": result[
                    "CalibrationError"
                ],
                "TheoryScore": result["TheoryScore"],
                "TrustScore": result["TrustScore"],
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    audit_df.to_csv(config.AUDIT_RESULTS, index=False)

    trust_df = audit_df[["PredictionID", "TrustScore"]]

    trust_df.to_csv(config.TRUST_TABLE, index=False)

    logger.info(
        "Audit finished: %d audited, %d skipped (no ground truth)",
        len(audit_df),
        skipped,
    )

    print(f"Audited predictions : {len(audit_df)}")
    print(f"Skipped (no GT)     : {skipped}")

    if not audit_df.empty:
        print()
        print(
            audit_df["TrustScore"].describe().round(2)
        )

    print()
    print("Audit finished.")

    return audit_df


# Step 4 : Evaluation & Backtesting

@execution_timer
def run_evaluation() -> pd.DataFrame:
    banner("STEP 4/5 : Evaluation & Backtesting")

    if not config.GT_TABLE.exists():
        logger.info(
            "GT table missing - running audit step first."
        )
        run_audit()

    gt = pd.read_csv(config.GT_TABLE)

    correct = gt["Correct"].dropna()

    if correct.empty:
        logger.warning("No ground-truth rows to evaluate.")
        return pd.DataFrame()

    # Directional hit rate

    hit_rate = correct.astype(bool).mean()

    print(f"Directional hit rate : {hit_rate:.2%}")

    # Classification metrics for the BUY signal

    market_up = (
        gt["ActualReturn"].dropna() > 0
    ).astype(int)

    llm_buy = (
        gt.loc[market_up.index, "Recommendation"] == "BUY"
    ).astype(int)

    cls_metrics = EvaluationMetrics.classification(
        market_up,
        llm_buy,
    )

    print()
    print(
        "LLM BUY signal vs market direction: "
        + ", ".join(
            f"{k}={v:.4f}"
            for k, v in cls_metrics.items()
        )
    )

    # Calibration

    mask = gt["Correct"].notna()

    ece = Calibration.expected_calibration_error(
        gt.loc[mask, "Confidence"],
        gt.loc[mask, "Correct"].astype(float),
    )

    print(f"Expected Calib. Error: {ece:.4f}")

    # Per-ticker backtest

    loader = DataLoader(config)

    prices = loader.load_market_prices()

    signal_map = {
        "BUY": 1,
        "SELL": -1,
        "HOLD": 0,
    }

    gt_signals = gt.set_index(["Date", "Ticker"])[
        "Recommendation"
    ].map(signal_map)

    backtests = {}

    for ticker, grp in prices.groupby("Ticker"):

        p = grp.sort_values("Date").reset_index(drop=True)

        p["Signal"] = p.set_index(["Date", "Ticker"]).index.map(
            gt_signals
        )

        bt = Backtester().run(p, p["Signal"])

        returns = bt["StrategyReturn"].dropna()

        if len(returns) < 2:
            continue

        backtests[ticker] = EvaluationMetrics.finance(
            returns
        )

    fin_df = pd.DataFrame(backtests).T

    print()
    print("Per-ticker strategy backtest:")
    print(fin_df.round(4))

    print()
    print("Evaluation complete.")

    return fin_df


# Step 5 : Explainability

@execution_timer
def run_explain() -> dict:
    banner("STEP 5/5 : Explainability")

    prices = _load_prices()

    row = prices.iloc[0]

    feature_names = SESSION.get("xgb_features") or [
        "Returns",
        "RSI",
        "MACD",
        "Sentiment",
    ]

    X = pd.DataFrame(
        [
            {
                col: float(row[col])
                for col in feature_names
                if col in row
            }
        ]
    )

    if "Sentiment" in feature_names and "Sentiment" not in X.columns:
        X["Sentiment"] = 0.6

    reasoning = SESSION.get(
        "llm_reasoning",
        "Strong momentum with positive sentiment "
        "and healthy RSI suggests upside.",
    )

    shap_explainer = None
    lime_explainer = None

    model = SESSION.get("xgb_model")

    if model is not None:
        try:
            from explainability import SHAPExplainer

            shap_explainer = SHAPExplainer(model)
        except Exception as exc:
            logger.info("SHAP unavailable: %s", exc)

        try:
            from explainability import LIMEExplainer

            train_data = SESSION.get("xgb_train_data")

            if train_data is not None:
                lime_explainer = LIMEExplainer(
                    train_data.values,
                    list(train_data.columns),
                )
        except Exception as exc:
            logger.info("LIME unavailable: %s", exc)

    pipeline = ExplainabilityPipeline(
        shap_explainer=shap_explainer,
        lime_explainer=lime_explainer,
    )

    result = pipeline.explain(
        model,
        X,
        reasoning,
    )

    print(json.dumps(result, default=str, indent=2))

    print()
    print("Explainability finished.")

    return result


# Dashboard

def launch_dashboard() -> None:
    banner("Launching Dashboard")

    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
        ]
    )


# Run Everything

def run_all(epochs: int = 3) -> None:
    run_pipeline()
    train_models(epochs=epochs)
    run_audit()
    run_evaluation()
    run_explain()

    print()
    print("=" * 60)
    print("Project finished successfully.")
    print("=" * 60)



# CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auditing LLM Trading - "
            "Bridging Theory and Market Reality "
            "with the GT Table"
        )
    )

    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run the data pipeline and build the merged dataset.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train XGBoost, LSTM and Transformer baselines.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Generate the GT table and run the audit engine.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Backtest, calibrate and evaluate the GT table.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run the explainability pipeline.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full workflow (steps 1-5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs for LSTM / Transformer training (default: %(default)s).",
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    set_random_seed(args.seed)

    ran = False

    if args.pipeline:
        run_pipeline()
        ran = True

    if args.train:
        train_models(epochs=args.epochs)
        ran = True

    if args.audit:
        run_audit()
        ran = True

    if args.evaluate:
        run_evaluation()
        ran = True

    if args.explain:
        run_explain()
        ran = True

    if args.dashboard:
        launch_dashboard()
        ran = True

    if args.all:
        run_all(epochs=args.epochs)
        ran = True

    if not ran:
        parser.print_help()


if __name__ == "__main__":
    main()
