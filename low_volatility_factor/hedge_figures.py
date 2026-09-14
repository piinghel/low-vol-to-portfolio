"""Render the article's rebalance-date hedge comparison from saved results."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from . import config, metrics, plot_diagnostics, plot_performance, plot_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hedge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = pl.read_parquet(args.baseline / "stage_daily.parquet").select(
        "date", "scenario", "net_return"
    )
    hedge = (
        pl.read_parquet(args.hedge / "daily.parquet")
        .filter(pl.col("scenario") == "naive_equal_ls")
        .select(
            "date",
            pl.lit("equal_weight_beta_hedged").alias("scenario"),
            (
                pl.col("net_return")
                + pl.col("hedge_gross_return")
                - pl.col("hedge_cost")
            ).alias("net_return"),
        )
        .sort("date")
    )
    saved = pl.read_csv(args.hedge / "metrics.csv")
    expected = saved.filter(
        (pl.col("scenario") == "naive_equal_ls") & (pl.col("funding_rate") == 0.0)
    ).row(0, named=True)
    for key, value in metrics._metric_row(
        hedge["net_return"].to_numpy(), annualization=252
    ).items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12)
    plotted = pl.concat([original, hedge])
    deciles = pl.read_csv(args.baseline / "decile_metrics.csv")
    for theme, suffix in (
        (config.PlotConfig(), ""),
        (plot_style.dark_plot_config(config.PlotConfig()), "_dark"),
    ):
        plot_style.render_figure(
            args.output,
            "decile_profile",
            partial(plot_diagnostics.plot_decile_profile, deciles, plot_config=theme),
            variant_suffix=suffix,
        )
        for mobile in (False, True):
            plot_style.render_figure(
                args.output,
                "performance_and_drawdowns" + ("_mobile" if mobile else ""),
                partial(
                    plot_performance.plot_performance_and_drawdowns,
                    plotted,
                    scenario_config=config.ScenarioConfig(),
                    plot_config=theme,
                    mobile=mobile,
                    include_hedge=True,
                ),
                variant_suffix=suffix,
            )
    manifest = {
        "method": "Saved point-in-time Russell 1000 hedge, rebalanced with stocks; 5bp trading costs; no funding charge. No new experiment or fitted full-sample hedge.",
        "checks": "Plotted hedge metrics reconcile to the previously completed zero-funding case.",
        "inputs": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.baseline / "stage_daily.parquet",
                args.hedge / "daily.parquet",
                args.hedge / "metrics.csv",
            )
        },
        "hedged_equal_metrics": expected,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
