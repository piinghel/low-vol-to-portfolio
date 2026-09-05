"""Explicit orchestration for the low-volatility article figure set."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl

from .config import BetaConfig, DataConfig, PlotConfig, ResearchConfig, ScenarioConfig
from .plot_diagnostics import (
    plot_beta_diagnostic,
    plot_decile_profile,
    plot_naive_leg_risk,
    plot_realized_exposures,
)
from .plot_performance import plot_performance_and_drawdowns, plot_regime_comparison
from .plot_style import dark_plot_config, render_figure


def render_article_figures(
    figures: Path,
    *,
    decile_metrics: pl.DataFrame,
    stage_metrics: pl.DataFrame,
    stage_daily: pl.DataFrame,
    scaled_leg_daily: pl.DataFrame,
    config: ResearchConfig,
) -> None:
    """Render every article figure from one explicit output specification."""

    def figure_renderers(plot_config: PlotConfig) -> dict[str, Callable[..., None]]:
        return {
            "decile_profile": partial(
                plot_decile_profile,
                decile_metrics,
                plot_config=plot_config,
            ),
            "naive_leg_risk": partial(
                plot_naive_leg_risk,
                stage_metrics,
                scenario_config=config.scenarios,
                plot_config=plot_config,
            ),
            "target_exposures": partial(
                plot_realized_exposures,
                stage_daily,
                scenario_config=config.scenarios,
                plot_config=plot_config,
            ),
            "beta_diagnostic": partial(
                plot_beta_diagnostic,
                stage_daily,
                scenario_config=config.scenarios,
                beta_config=config.beta,
                plot_config=plot_config,
            ),
            "performance_and_drawdowns": partial(
                plot_performance_and_drawdowns,
                stage_daily,
                scenario_config=config.scenarios,
                plot_config=plot_config,
            ),
            "regime_comparison": partial(
                plot_regime_comparison,
                stage_daily,
                scaled_leg_daily,
                scenario_config=config.scenarios,
                plot_config=plot_config,
            ),
        }

    variants = (
        (config.plots, ""),
        (dark_plot_config(config.plots), "_dark"),
    )
    for plot_config, suffix in variants:
        for stem, renderer in figure_renderers(plot_config).items():
            render_figure(figures, stem, renderer, variant_suffix=suffix)
            if stem in {
                "naive_leg_risk",
                "performance_and_drawdowns",
                "regime_comparison",
            }:
                render_figure(
                    figures,
                    stem + "_mobile",
                    partial(renderer, mobile=True),
                    variant_suffix=suffix,
                )


def main() -> None:
    """Rebuild displays from compact saved results, without loading stock data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads((args.input / "config.json").read_text())
    # Renderers use these saved settings; no signal or backtest is constructed.
    config = ResearchConfig(
        data=DataConfig(**saved["data"]),
        scenarios=ScenarioConfig(**saved["scenarios"]),
        plots=PlotConfig(**saved["plots"]),
        beta=BetaConfig(**saved["beta"]),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    render_article_figures(
        args.output,
        decile_metrics=pl.scan_csv(args.input / "decile_metrics.csv").collect(),
        stage_metrics=pl.scan_csv(args.input / "stage_metrics.csv").collect(),
        stage_daily=pl.scan_parquet(args.input / "stage_daily.parquet").collect(),
        scaled_leg_daily=pl.scan_parquet(
            args.input / "scaled_leg_daily.parquet"
        ).collect(),
        config=config,
    )


if __name__ == "__main__":
    main()
