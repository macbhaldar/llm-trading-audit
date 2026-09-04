"""
Calibration Visualizations
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from .base import BaseVisualizer


class CalibrationVisualizer(BaseVisualizer):

    # Reliability Diagram
    def reliability_diagram(
        self,
        y_true,
        y_prob,
        n_bins=10,
        title="Reliability Diagram",
    ):

        prob_true, prob_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
            strategy="uniform",
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Model",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(
                    dash="dash",
                    color="gray",
                ),
                name="Perfect Calibration",
            )
        )

        fig.update_xaxes(
            title="Mean Predicted Probability"
        )

        fig.update_yaxes(
            title="Observed Frequency"
        )

        return self.apply_layout(fig, title)

    
    # Confidence Histogram
    
    def confidence_histogram(
        self,
        confidence,
        bins=20,
        title="Confidence Distribution",
    ):

        fig = px.histogram(
            x=confidence,
            nbins=bins,
        )

        fig.update_xaxes(
            title="Prediction Confidence"
        )

        return self.apply_layout(fig, title)

    
    # Confidence vs Accuracy
    
    def confidence_accuracy(
        self,
        df,
        confidence_col="Confidence",
        accuracy_col="Correct",
        title="Confidence vs Accuracy",
    ):

        self.validate_columns(
            df,
            [
                confidence_col,
                accuracy_col,
            ],
        )

        fig = px.scatter(
            df,
            x=confidence_col,
            y=accuracy_col,
            trendline="ols",
        )

        return self.apply_layout(fig, title)

    
    # Calibration Error Bar
    
    def calibration_error(
        self,
        models,
        ece,
        title="Expected Calibration Error",
    ):

        fig = px.bar(
            x=models,
            y=ece,
            labels={
                "x": "Model",
                "y": "ECE",
            },
        )

        return self.apply_layout(fig, title)

    
    # Brier Score Comparison
    
    def brier_scores(
        self,
        scores,
        title="Brier Score Comparison",
    ):

        df = pd.DataFrame(scores)

        fig = px.bar(
            df,
            x="Model",
            y="BrierScore",
            color="Model",
        )

        return self.apply_layout(fig, title)

    
    # Calibration by Model

    def multi_model_calibration(
        self,
        calibration_results,
        title="Calibration Comparison",
    ):

        fig = go.Figure()

        for model_name, values in calibration_results.items():
            prob_true = values["prob_true"]
            prob_pred = values["prob_pred"]

            fig.add_trace(
                go.Scatter(
                    x=prob_pred,
                    y=prob_true,
                    mode="lines+markers",
                    name=model_name,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(
                    dash="dash",
                    color="black",
                ),
                name="Perfect",
            )
        )
        return self.apply_layout(fig, title)

    
    # Confidence Density
    
    def confidence_density(
        self,
        confidence,
        title="Confidence Density",
    ):

        fig = px.density_contour(
            x=confidence,
            y=np.random.normal(
                size=len(confidence)
            ),
        )

        return self.apply_layout(fig, title)

    
    # Prediction Probability Boxplot
    
    def probability_boxplot(
        self,
        df,
        probability_col="Confidence",
        model_col="Model",
        title="Prediction Probability",
    ):

        self.validate_columns(
            df,
            [
                probability_col,
                model_col,
            ],
        )

        fig = px.box(
            df,
            x=model_col,
            y=probability_col,
            color=model_col,
        )

        return self.apply_layout(fig, title)

    
    # Probability Heatmap
    
    def probability_heatmap(
        self,
        df,
        index="Model",
        columns="Ticker",
        values="Confidence",
        title="Confidence Heatmap",
    ):

        table = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=np.mean,
        )

        fig = px.imshow(
            table,
            text_auto=".2f",
            aspect="auto",
        )

        return self.apply_layout(fig, title)

    
    # Probability Timeline
    
    def confidence_timeline(
        self,
        df,
        date_col="Date",
        confidence_col="Confidence",
        model_col="Model",
        title="Confidence Timeline",
    ):

        self.validate_columns(
            df,
            [
                date_col,
                confidence_col,
                model_col,
            ],
        )

        fig = px.line(
            df,
            x=date_col,
            y=confidence_col,
            color=model_col,
            markers=True,
        )

        return self.apply_layout(fig, title)

    
    # Compute Expected Calibration Error
    
    @staticmethod
    def expected_calibration_error(
        y_true,
        y_prob,
        n_bins=10,
    ):

        bins = np.linspace(0, 1, n_bins + 1)

        ids = np.digitize(
            y_prob,
            bins
        ) - 1

        ece = 0

        for i in range(n_bins):
            mask = ids == i
            if np.sum(mask) == 0:
                continue
            acc = np.mean(
                y_true[mask]
            )
            conf = np.mean(
                y_prob[mask]
            )
            ece += (
                np.abs(acc - conf)
                * np.sum(mask)
                / len(y_true)
            )
        return float(ece)

    
    # Maximum Calibration Error
    
    @staticmethod
    def maximum_calibration_error(
        y_true,
        y_prob,
        n_bins=10,
    ):

        bins = np.linspace(0, 1, n_bins + 1)
        ids = np.digitize(
            y_prob,
            bins
        ) - 1

        errors = []
        for i in range(n_bins):

            mask = ids == i

            if np.sum(mask) == 0:
                continue
            acc = np.mean(
                y_true[mask]
            )
            conf = np.mean(
                y_prob[mask]
            )
            errors.append(
                abs(acc - conf)
            )
        return float(max(errors))

    
    # Calibration Summary
    
    def calibration_summary(
        self,
        y_true,
        y_prob,
    ):

        return {
            "Brier Score":
                float(
                    brier_score_loss(
                        y_true,
                        y_prob,
                    )
                ),

            "ECE":
                self.expected_calibration_error(
                    y_true,
                    y_prob,
                ),

            "MCE":
                self.maximum_calibration_error(
                    y_true,
                    y_prob,
                ),

            "Average Confidence":
                float(
                    np.mean(
                        y_prob
                    )
                ),
        }
