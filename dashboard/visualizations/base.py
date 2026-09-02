"""
Base Visualization - Provides common Plotly configuration for all dashboard charts.
"""

from abc import ABC
import plotly.graph_objects as go

from dashboard.config import (
    PLOTLY_TEMPLATE,
    DEFAULT_HEIGHT,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SUCCESS_COLOR,
    ERROR_COLOR,
)

class BaseVisualizer(ABC):
    def __init__(self):
        self.template = PLOTLY_TEMPLATE
        self.default_height = DEFAULT_HEIGHT
        self.primary = PRIMARY_COLOR
        self.secondary = SECONDARY_COLOR
        self.success = SUCCESS_COLOR
        self.error = ERROR_COLOR

    def apply_layout(
        self,
        fig: go.Figure,
        title: str,
        height: int = None,
    ) -> go.Figure:

        fig.update_layout(
            template=self.template,
            title=title,
            height=height or self.default_height,
            hovermode="x unified",

            legend=dict(
                orientation="h",
                y=1.02,
                x=1,
                xanchor="right",
            ),

            margin=dict(
                l=40,
                r=30,
                t=60,
                b=40,
            ),
        )

        fig.update_xaxes(
            showgrid=True,
            gridwidth=0.5,
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=0.5,
        )
        return fig

    def create_empty_chart(
        self,
        title="No Data"
    ):

        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            showarrow=False,
            font=dict(
                size=18
            )
        )

        return self.apply_layout(
            fig,
            title,
        )

    @staticmethod
    def validate_columns(
        dataframe,
        required_columns,
    ):

        missing = [
            col
            for col in required_columns
            if col not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                "Missing columns: "
                + ", ".join(missing)
            )

        return True
    