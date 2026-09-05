"""Universe, signal, exposure, and beta diagnostics for the article."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FuncFormatter

from .config import BetaConfig, PlotConfig, ScenarioConfig
from .plot_style import clean_axis, finish_figure


def plot_decile_profile(
    metrics: pl.DataFrame,
    path: Path,
    plot_config: PlotConfig,
) -> None:
    """Plot return, risk, and Sharpe across volatility deciles."""

    ordered = metrics.with_columns(
        pl.col("scenario").str.extract(r"(\d+)$", 1).cast(pl.Int8).alias("decile")
    ).sort("decile")
    x = ordered.get_column("decile").to_list()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 8.0),
    )
    specs = [
        ("sharpe_ratio", "Sharpe ratio", 1.0),
        ("geometric_return", "Geometric return (%)", 100.0),
        ("volatility", "Volatility (%)", 100.0),
    ]
    for axis, (column, title, multiplier) in zip(axes, specs, strict=True):
        axis.bar(
            x,
            [value * multiplier for value in ordered.get_column(column)],
            color=[
                plot_config.low_volatility_color
                if decile == min(x)
                else plot_config.high_volatility_color
                if decile == max(x)
                else plot_config.decile_profile_color
                for decile in x
            ],
        )
        axis.set_title(
            title,
            loc="left",
            pad=9,
            color=plot_config.muted_text_color,
            fontweight="normal",
        )
        tick_labels = [
            "1\nLow-vol"
            if decile == min(x)
            else "10\nHigh-vol"
            if decile == max(x)
            else str(decile)
            for decile in x
        ]
        axis.set_xticks(x, tick_labels)
        clean_axis(axis, plot_config)
    finish_figure(figure, path, plot_config, axis_label_size=14.5)


def plot_naive_leg_risk(
    metrics: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    *,
    mobile: bool = False,
) -> None:
    """Compare the risk of equal-dollar low- and high-volatility legs."""

    selected = metrics.filter(
        (pl.col("fee_state") == "before_costs")
        & pl.col("scenario").is_in(
            [
                scenario_config.low_volatility_long,
                scenario_config.high_volatility_long,
            ]
        )
    ).sort("scenario")
    scenarios = [
        scenario_config.low_volatility_long,
        scenario_config.high_volatility_long,
    ]
    labels = {
        scenario_config.low_volatility_long: "Low-volatility decile",
        scenario_config.high_volatility_long: "High-volatility decile",
    }
    colors = {
        scenario_config.low_volatility_long: plot_config.low_volatility_color,
        scenario_config.high_volatility_long: plot_config.high_volatility_color,
    }
    volatility_values = [
        float(selected.filter(pl.col("scenario") == scenario)["volatility"][0]) * 100
        for scenario in scenarios
    ]
    beta_values = [
        float(
            selected.filter(pl.col("scenario") == scenario)[
                "average_ex_ante_stock_beta"
            ][0]
        )
        for scenario in scenarios
    ]
    figure, axes = plt.subplots(
        2 if mobile else 1,
        1 if mobile else 2,
        figsize=(4.8, 4.8) if mobile else (9.0, 2.8),
    )
    panel_specs = [
        (volatility_values, "Annualized volatility", "%.1f%%"),
        (beta_values, "Average ex-ante beta", "%.2f"),
    ]
    bar_positions = [0.46, 0.54]
    for index, (axis, (values, title, label_format)) in enumerate(
        zip(axes, panel_specs, strict=True)
    ):
        bars = axis.barh(
            bar_positions,
            values,
            color=[colors[scenario] for scenario in scenarios],
            height=0.05,
        )
        axis.set_yticks(bar_positions, [labels[scenario] for scenario in scenarios])
        axis.set_ylim(0.43, 0.57)
        axis.bar_label(
            bars,
            fmt=label_format,
            padding=3,
            color=plot_config.muted_text_color,
            fontsize=10.6,
            fontweight="regular",
            alpha=0.75,
        )
        axis.set_xlabel(title)
        axis.set_xlim(0, max(values) * 1.2)
        clean_axis(axis, plot_config)
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", color=plot_config.grid_color, linewidth=0.8)
        if index == 1 and not mobile:
            axis.tick_params(axis="y", left=False, labelleft=False)
    finish_figure(figure, path, plot_config)


def plot_realized_exposures(
    daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
) -> None:
    """Plot realized floating long, short, and net stock exposure."""

    data = daily.filter(
        pl.col("scenario") == scenario_config.volatility_scaled_long_short
    ).sort("signal_date")
    dates = data.get_column("date").to_list()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 5.6),
        sharex=True,
    )
    panels = [
        ("long_exposure", "Long gross", plot_config.low_volatility_color),
        ("short_exposure", "Short gross", plot_config.high_volatility_color),
        ("net_exposure", "Net exposure", plot_config.volatility_scaled_color),
    ]
    for axis, (column, label, color) in zip(axes, panels, strict=True):
        axis.plot(dates, data.get_column(column), color=color, linewidth=1.35)
        axis.set_ylim(0, 1.12)
        axis.set_yticks([0, 0.5, 1.0])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        axis.set_title(
            label,
            loc="left",
            pad=9,
            color=plot_config.muted_text_color,
            fontweight="normal",
        )
        clean_axis(axis, plot_config)
    finish_figure(figure, path, plot_config)


def plot_beta_diagnostic(
    daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    beta_config: BetaConfig,
    plot_config: PlotConfig,
) -> None:
    """Plot ex-ante stock beta beside rolling realized market beta."""

    data = (
        daily.filter(pl.col("scenario") == scenario_config.volatility_scaled_long_short)
        .sort("date")
        .with_columns(
            pl.rolling_cov(
                pl.col("gross_return"),
                pl.col("market_return"),
                window_size=beta_config.lookback,
                min_samples=beta_config.minimum_observations,
            ).alias("rolling_market_covariance"),
            pl.col("market_return")
            .rolling_var(
                window_size=beta_config.lookback,
                min_samples=beta_config.minimum_observations,
            )
            .alias("rolling_market_variance"),
        )
        .with_columns(
            (
                pl.col("rolling_market_covariance") / pl.col("rolling_market_variance")
            ).alias("rolling_realized_beta")
        )
    )
    dates = data.get_column("date").to_list()
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        dates,
        data.get_column("stock_beta"),
        label="Ex-ante stock beta",
        color=plot_config.ex_ante_beta_color,
        linewidth=1.15,
        linestyle=(0, (3, 2)),
        alpha=0.75,
    )
    axis.plot(
        dates,
        data.get_column("rolling_realized_beta"),
        label=f"Realized beta ({beta_config.lookback} days)",
        color=plot_config.realized_beta_color,
        linewidth=1.7,
    )
    axis.set_title(
        "Portfolio beta",
        loc="left",
        pad=9,
        color=plot_config.muted_text_color,
        fontweight="normal",
    )
    axis.legend(frameon=False, ncol=2)
    clean_axis(axis, plot_config)
    finish_figure(figure, path, plot_config)
