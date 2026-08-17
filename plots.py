"""Deterministic figures for the low-volatility article."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FuncFormatter, NullFormatter

from .config import BetaConfig, PlotConfig, ResearchConfig, ScenarioConfig
from .frames import require_finite_float
from .metrics import cumulative_returns, drawdown_series


def _finish_figure(fig: plt.Figure, path: Path, plot_config: PlotConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(plot_config.background_color)
    for axis in fig.axes:
        axis.set_facecolor(plot_config.background_color)
        axis.tick_params(colors=plot_config.muted_text_color, labelsize=10.5)
        axis.xaxis.label.set_color(plot_config.muted_text_color)
        axis.xaxis.label.set_fontsize(11.5)
        axis.yaxis.label.set_color(plot_config.muted_text_color)
        axis.yaxis.label.set_fontsize(11.5)
        axis.title.set_color(plot_config.text_color)
        axis.title.set_fontweight("bold")
        axis.title.set_fontsize(12)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_color(plot_config.text_color)
                label.set_fontsize(10.5)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor=plot_config.background_color,
    )
    plt.close(fig)


def _clean_axis(axis: plt.Axes, plot_config: PlotConfig) -> None:
    axis.spines["left"].set_color(plot_config.grid_color)
    axis.spines["bottom"].set_color(plot_config.grid_color)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        axis="y",
        color=plot_config.grid_color,
        linewidth=0.7,
        alpha=0.8,
    )
    axis.set_axisbelow(True)
    axis.margins(x=0)


def plot_eligible_universe(
    snapshot_sizes: pl.DataFrame,
    path: Path,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    """Plot the point-in-time eligible cross-section at each signal date."""
    data = snapshot_sizes.sort("date")
    dates = data.get_column("date").to_list()
    counts = data.get_column("eligible_stocks")
    minimum = int(require_finite_float(counts.min(), "minimum eligible stocks"))
    median = require_finite_float(counts.median(), "median eligible stocks")
    maximum = int(require_finite_float(counts.max(), "maximum eligible stocks"))

    fig, axis = plt.subplots(figsize=(7.0, 5.2) if mobile_layout else (9, 4.2))
    axis.plot(
        dates,
        counts,
        color=plot_config.low_volatility_color,
        linewidth=1.35,
    )
    axis.axhline(
        median,
        color=plot_config.zero_line_color,
        linewidth=1.1,
        linestyle="--",
        label=f"Median ({median:,.0f})",
    )
    axis.set_xlim(dates[0], dates[-1])
    axis.set_ylim(minimum, maximum)
    axis.legend(frameon=False)
    _clean_axis(axis, plot_config)
    _finish_figure(fig, path, plot_config)


def plot_decile_profile(
    metrics: pl.DataFrame,
    path: Path,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    ordered = metrics.with_columns(
        pl.col("scenario").str.extract(r"(\d+)$", 1).cast(pl.Int8).alias("decile")
    ).sort("decile")
    x = ordered.get_column("decile").to_list()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.4, 10.0) if mobile_layout else (9, 8.0),
    )
    specs = [
        ("geometric_return", "Geometric return (%)", 100.0),
        ("volatility", "Volatility (%)", 100.0),
        ("sharpe_ratio", "Sharpe ratio", 1.0),
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
        axis.set_ylabel(title)
        axis.set_xticks(x)
        _clean_axis(axis, plot_config)
    axes[-1].set_xlabel("Volatility decile")
    _finish_figure(fig, path, plot_config)


def plot_naive_leg_risk(
    metrics: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
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
    if mobile_layout:
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panel_specs = [
        (volatility_values, "Annualized volatility", "%.1f%%"),
        (beta_values, "Average ex-ante beta", "%.2f"),
    ]
    for index, (axis, (values, title, label_format)) in enumerate(
        zip(axes, panel_specs, strict=True)
    ):
        bars = axis.barh(
            [labels[scenario] for scenario in scenarios],
            values,
            color=[colors[scenario] for scenario in scenarios],
            height=0.56,
        )
        axis.bar_label(
            bars,
            fmt=label_format,
            padding=5,
            color=plot_config.text_color,
            fontweight="bold",
        )
        axis.set_xlabel(title)
        axis.set_xlim(0, max(values) * 1.2)
        _clean_axis(axis, plot_config)
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", color=plot_config.grid_color, linewidth=0.7, alpha=0.8)
        if index == 1 and not mobile_layout:
            axis.tick_params(axis="y", left=False, labelleft=False)
    _finish_figure(fig, path, plot_config)


def plot_target_exposures(
    target_exposures: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    data = target_exposures.filter(
        pl.col("scenario") == scenario_config.volatility_scaled_long_short
    ).sort("signal_date")
    dates = data.get_column("signal_date").to_list()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.0, 7.4) if mobile_layout else (9, 5.6),
        sharex=True,
    )
    panels = [
        ("long_exposure", "Long gross", plot_config.low_volatility_color),
        ("short_exposure", "Short gross", plot_config.high_volatility_color),
        ("net_exposure", "Net exposure", plot_config.volatility_scaled_color),
    ]
    for axis, (column, label, color) in zip(axes, panels, strict=True):
        axis.plot(dates, data.get_column(column), color=color, linewidth=1.35)
        axis.set_ylim(0, 1.05)
        axis.set_yticks([0, 0.5, 1.0])
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.set_title(label, loc="left", pad=9)
        _clean_axis(axis, plot_config)
    _finish_figure(fig, path, plot_config)


def plot_beta_diagnostic(
    daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    beta_config: BetaConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
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
    fig, axis = plt.subplots(figsize=(7.0, 6.0) if mobile_layout else (9, 4.5))
    axis.plot(
        dates,
        data.get_column("stock_beta"),
        label="Aggregated ex-ante stock beta",
        color=plot_config.volatility_scaled_color,
        alpha=0.75,
    )
    axis.plot(
        dates,
        data.get_column("rolling_realized_beta"),
        label=f"{beta_config.lookback}-day realized beta",
        color=plot_config.realized_beta_color,
    )
    axis.axhspan(-0.1, 0.1, color=plot_config.grid_color, alpha=0.3, linewidth=0)
    axis.axhline(0, color=plot_config.zero_line_color, linewidth=0.8)
    axis.set_ylabel("Beta")
    axis.legend(frameon=False, ncol=1 if mobile_layout else 2)
    _clean_axis(axis, plot_config)
    _finish_figure(fig, path, plot_config)


def plot_cumulative_performance(
    daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    scenarios, labels, colors = _comparison_style(scenario_config, plot_config)
    data = cumulative_returns(
        daily.filter(pl.col("scenario").is_in(scenarios)), return_column="net_return"
    )
    fig, axis = plt.subplots(figsize=(7.0, 6.0) if mobile_layout else (9, 4.8))
    for scenario in scenarios:
        part = data.filter(pl.col("scenario") == scenario).sort("date")
        axis.plot(
            part.get_column("date").to_list(),
            part.get_column("wealth"),
            label=labels[scenario],
            color=colors[scenario],
        )
    axis.axhline(1, color=plot_config.zero_line_color, linewidth=0.8)
    axis.set_yscale("log")
    wealth_min = require_finite_float(
        data.get_column("wealth").min(), "minimum cumulative wealth"
    )
    wealth_max = require_finite_float(
        data.get_column("wealth").max(), "maximum cumulative wealth"
    )
    candidate_ticks = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    visible_ticks = [
        tick for tick in candidate_ticks if wealth_min * 0.9 <= tick <= wealth_max * 1.1
    ]
    axis.set_yticks(visible_ticks)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_ylabel("Wealth (base = 1, log scale)")
    axis.legend(frameon=False)
    _clean_axis(axis, plot_config)
    _finish_figure(fig, path, plot_config)


def plot_drawdowns(
    daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    scenarios, labels, colors = _comparison_style(scenario_config, plot_config)
    data = drawdown_series(
        daily.filter(pl.col("scenario").is_in(scenarios)), return_column="net_return"
    )
    fig, axis = plt.subplots(figsize=(7.0, 6.0) if mobile_layout else (9, 4.5))
    for scenario in scenarios:
        part = data.filter(pl.col("scenario") == scenario).sort("date")
        maximum_drawdown = require_finite_float(
            part.get_column("drawdown").min(), "maximum drawdown"
        )
        axis.plot(
            part.get_column("date").to_list(),
            part.get_column("drawdown") * 100,
            label=f"{labels[scenario]} ({maximum_drawdown:.0%})",
            color=colors[scenario],
        )
    axis.set_ylabel("Drawdown (%)")
    axis.legend(frameon=False)
    _clean_axis(axis, plot_config)
    _finish_figure(fig, path, plot_config)


def _comparison_style(
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return the shared labels and colors for the two L/S implementations."""

    scenarios = [
        scenario_config.naive_equal_weight_long_short,
        scenario_config.volatility_scaled_long_short,
    ]
    labels = {
        scenario_config.naive_equal_weight_long_short: ("Equal-weight reference"),
        scenario_config.volatility_scaled_long_short: ("Stock-volatility-scaled L/S"),
    }
    colors = {
        scenario_config.naive_equal_weight_long_short: (
            plot_config.naive_long_short_color
        ),
        scenario_config.volatility_scaled_long_short: (
            plot_config.volatility_scaled_color
        ),
    }
    return scenarios, labels, colors


def _render_layout_pair(
    figures: Path,
    stem: str,
    renderer: Callable[..., None],
) -> None:
    """Render one desktop/mobile pair with stable article filenames."""

    renderer(path=figures / f"{stem}.png", mobile_layout=False)
    renderer(path=figures / f"{stem}_mobile.png", mobile_layout=True)


def render_article_figures(
    figures: Path,
    *,
    snapshot_sizes: pl.DataFrame,
    decile_metrics: pl.DataFrame,
    stage_metrics: pl.DataFrame,
    target_exposures: pl.DataFrame,
    stage_daily: pl.DataFrame,
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
            plot_target_exposures,
            target_exposures,
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
        "cumulative_performance": partial(
            plot_cumulative_performance,
            stage_daily,
            scenario_config=config.scenarios,
            plot_config=config.plots,
        ),
        "drawdowns": partial(
            plot_drawdowns,
            stage_daily,
            scenario_config=config.scenarios,
            plot_config=config.plots,
        ),
    }
    for stem, renderer in renderers.items():
        _render_layout_pair(figures, stem, renderer)
