# Main Entry Point

import argparse
import subprocess
import sys
from pathlib import Path


# Project Root

ROOT = Path(__file__).resolve().parent


# Import Project Modules

from data_pipeline import DataPipeline

from auditing import AuditPipeline

from evaluation import EvaluationMetrics


# Data Pipeline

def run_pipeline():

    print("=" * 60)
    print("Running Data Pipeline...")
    print("=" * 60)

    pipeline = DataPipeline()

    df = pipeline.run()

    print(df.head())

    print()

    print("Pipeline Finished")

    return df



# Train Models

def train_models():

    print("=" * 60)
    print("Training Models")
    print("=" * 60)
    print("Training Transformer...")

    # TODO
    # transformer trainer
    print("Training LSTM...")

    # TODO
    print("Training XGBoost...")

    # TODO
    print("Training LLM...")

    # TODO
    print()
    print("Training Complete")



# Evaluation

def evaluate():

    print("=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    # placeholder
    print("Evaluation Complete")


# Auditing

def audit():

    print("=" * 60)
    print("Running Audit")
    print("=" * 60)

    pipeline = AuditPipeline()

    print("Audit Finished")

    return pipeline


# Explainability

def explain():

    print("=" * 60)
    print("Generating Explainability")
    print("=" * 60)
    print("SHAP...")
    print("LIME...")
    print("Attention...")
    print("Counterfactual...")
    print("Finished")


# Launch Dashboard

def dashboard():

    print("=" * 60)
    print("Launching Dashboard")
    print("=" * 60)

    dashboard_path = ROOT / "dashboard" / "app.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path)
        ]
    )


# Run Everything

def run_all():
    run_pipeline()
    train_models()
    evaluate()
    audit()
    explain()
    print()
    print("=" * 60)
    print("Project Finished Successfully")
    print("=" * 60)


# CLI

def build_parser():
    parser = argparse.ArgumentParser(
        description="Auditing LLM Trading"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true"
    )
    parser.add_argument(
        "--train",
        action="store_true"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true"
    )

    parser.add_argument(
        "--audit",
        action="store_true"
    )

    parser.add_argument(
        "--explain",
        action="store_true"
    )

    parser.add_argument(
        "--dashboard",
        action="store_true"
    )

    parser.add_argument(
        "--all",
        action="store_true"
    )
    return parser


# Main

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.pipeline:
        run_pipeline()

    elif args.train:
        train_models()

    elif args.evaluate:
        evaluate()

    elif args.audit:
        audit()

    elif args.explain:
        explain()

    elif args.dashboard:
        dashboard()

    elif args.all:
        run_all()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
