from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_sensitivity_figure(results_df, parameter: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Revenue Distribution Drift",
            "Price Elasticity Estimate",
            "Weekend Effect Size",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=results_df["value"],
            y=results_df["revenue_distribution_shift"],
            mode="lines+markers",
            name="Drift",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=results_df["value"],
            y=results_df["price_elasticity_estimate"],
            mode="lines+markers",
            name="Elasticity",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=results_df["value"],
            y=results_df["weekend_effect_size"],
            mode="lines+markers",
            name="Weekend effect",
        ),
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text=parameter, row=3, col=1)
    fig.update_yaxes(title_text="Wasserstein distance", row=1, col=1)
    fig.update_yaxes(title_text="log-log slope", row=2, col=1)
    fig.update_yaxes(title_text="Cohen's d", row=3, col=1)
    fig.update_layout(height=900, title_text=f"Sensitivity sweep for '{parameter}'", showlegend=False)
    return fig
