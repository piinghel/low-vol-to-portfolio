"""Explicit orchestration for the low-volatility article figure set."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl

from .config import ResearchConfig
from .plot_diagnostics import (
    plot_beta_diagnostic,
    plot_decile_profile,
    plot_eligible_universe,
    plot_naive_leg_risk,
    plot_realized_exposures,
)
from .plot_performance import plot_performance_and_drawdowns, plot_regime_comparison
from .plot_style import render_layout_pair


def render_article_figures(
    figures: Path,
    *,
    snapshot_sizes: pl.DataFrame,
    decile_metrics: pl.DataFrame,
    stage_metrics: pl.DataFrame,
    stage_daily: pl.DataFrame,
    scaled_leg_daily: pl.DataFrame,
    config: ResearchConfig,
) -> None:
    """Render every article figure from one explicit output specification."""

    renderers: dict[str, Callable[..., None]] = {
        "eligible_universe": partial(
            plot_eligible_universe,
            snapshot_sizes,
            plot_config=config.plots,
        ),
        "decile_profile": partial(
            plot_decile_profile,
            decile_metrics,
            plot_config=config.plots,
        ),
        "naive_leg_risk": partial(
            plot_naive_leg_risk,
            stage_metrics,
            scenario_config=config.scenarios,
            plot_config=config.plots,
        ),
        "target_exposures": partial(
            plot_realized_exposures,
            stage_daily,
            scenario_config=config.scenarios,
            plot_config=config.plots,
        ),
        "beta_diagnostic": partial(
            plot_beta_diagnostic,
            stage_daily,
            scenario_config=config.scenarios,
            beta_config=config.beta,
            plot_config=config.plots,
        ),
        "performance_and_drawdowns": partial(
            plot_performance_and_drawdowns,
            stage_daily,
            scenario_config=config.scenarios,
            plot_config=config.plots,
        ),
        "regime_comparison": partial(
            plot_regime_comparison,
            stage_daily,
            scaled_leg_daily,
            scenario_config=config.scenarios,
            plot_config=config.plots,
        ),
    }
    for stem, renderer in renderers.items():
        render_layout_pair(figures, stem, renderer)
