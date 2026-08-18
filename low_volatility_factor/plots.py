"""Deterministic figures for the low-volatility article."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

from .config import BetaConfig, PlotConfig, ResearchConfig, ScenarioConfig
from .frames import require_finite_float
from .metrics import cumulative_returns, drawdown_series


def _finish_figure(
    fig: plt.Figure,
    path: Path,
    plot_config: PlotConfig,
    *,
    tight_layout: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(plot_config.background_color)
    for axis in fig.axes:
        axis.set_facecolor(plot_config.background_color)
        axis.tick_params(
            colors=plot_config.muted_text_color,
            labelsize=10.5,
            length=0,
            width=0,
        )
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
    if tight_layout:
        fig.tight_layout()
    fig.savefig(
        path,
        format=path.suffix.removeprefix("."),
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
        linestyle=":",
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
        scenario_config.low_volatility_long: plot_config.reference_low_color,
        scenario_config.high_volatility_long: plot_config.reference_high_color,
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
        axis.set_ylim(0.38, 0.62)
        axis.bar_label(
            bars,
            fmt=label_format,
            padding=3,
            color=plot_config.muted_text_color,
            fontsize=8.5,
            fontweight="regular",
            alpha=0.75,
        )
        axis.set_xlabel(title)
        axis.set_xlim(0, max(values) * 1.2)
        _clean_axis(axis, plot_config)
        axis.grid(axis="y", visible=False)
        axis.grid(
            axis="x",
            color=plot_config.grid_color,
            linewidth=0.7,
            linestyle=":",
            alpha=0.8,
        )
        if index == 1 and not mobile_layout:
            axis.tick_params(axis="y", left=False, labelleft=False)
    _finish_figure(fig, path, plot_config)


def plot_realized_exposures(
    daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    data = daily.filter(
        pl.col("scenario") == scenario_config.volatility_scaled_long_short
    ).sort("signal_date")
    dates = data.get_column("date").to_list()
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
        axis.set_ylim(0, 1.12)
        axis.set_yticks([0, 0.5, 1.0])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
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
        color=plot_config.ex_ante_beta_color,
        linewidth=1.1,
        alpha=0.85,
    )
    axis.plot(
        dates,
        data.get_column("rolling_realized_beta"),
        label=f"{beta_config.lookback}-day realized beta",
        color=plot_config.realized_beta_color,
        linewidth=1.6,
    )
    axis.axhspan(-0.1, 0.1, color=plot_config.grid_color, alpha=0.3, linewidth=0)
    axis.axhline(
        0,
        color=plot_config.zero_line_color,
        linewidth=0.8,
        linestyle=":",
    )
    axis.set_ylabel("Beta")
    axis.legend(frameon=False, ncol=1 if mobile_layout else 2)
    _clean_axis(axis, plot_config)
    _finish_figure(fig, path, plot_config)


def plot_performance_and_drawdowns(
    daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    """Plot cumulative wealth and drawdowns in one aligned figure."""

    scenarios, labels, colors = _comparison_style(scenario_config, plot_config)
    filtered = daily.filter(pl.col("scenario").is_in(scenarios))
    performance = cumulative_returns(filtered, return_column="net_return")
    drawdowns = drawdown_series(filtered, return_column="net_return")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 8.6) if mobile_layout else (9.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
    )
    wealth_axis, drawdown_axis = axes
    for scenario in scenarios:
        wealth = performance.filter(pl.col("scenario") == scenario).sort("date")
        wealth_axis.plot(
            wealth.get_column("date").to_list(),
            wealth.get_column("wealth"),
            label=labels[scenario],
            color=colors[scenario],
        )
        drawdown = drawdowns.filter(pl.col("scenario") == scenario).sort("date")
        dates = drawdown.get_column("date").to_list()
        values = drawdown.get_column("drawdown") * 100
        drawdown_axis.plot(dates, values, color=colors[scenario])
        drawdown_axis.fill_between(
            dates,
            values,
            0,
            color=colors[scenario],
            alpha=0.07,
            linewidth=0,
        )
    wealth_axis.axhline(
        1,
        color=plot_config.zero_line_color,
        linewidth=0.8,
        linestyle=":",
    )
    wealth_axis.set_yscale("log")
    wealth_min = require_finite_float(
        performance.get_column("wealth").min(), "minimum cumulative wealth"
    )
    wealth_max = require_finite_float(
        performance.get_column("wealth").max(), "maximum cumulative wealth"
    )
    candidate_ticks = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    wealth_axis.set_yticks(
        [
            tick
            for tick in candidate_ticks
            if wealth_min * 0.9 <= tick <= wealth_max * 1.1
        ]
    )
    wealth_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    wealth_axis.yaxis.set_minor_formatter(NullFormatter())
    wealth_axis.yaxis.set_minor_locator(NullLocator())
    wealth_axis.set_ylabel("Wealth (base = 1, log scale)")
    wealth_axis.legend(frameon=False)
    drawdown_axis.set_ylabel("Drawdown (%)")
    drawdown_axis.set_xlabel("Date")
    for axis in axes:
        _clean_axis(axis, plot_config)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.09, top=0.98, hspace=0.08)
    _finish_figure(fig, path, plot_config, tight_layout=False)


def _compound_return_series(
    frame: pl.DataFrame,
    return_column: str,
) -> pl.DataFrame:
    return frame.sort("date").select(
        "date",
        (1.0 + pl.col(return_column).fill_null(0.0)).cum_prod().alias("wealth"),
    )


def _compound_leg_contribution(
    combined: pl.DataFrame,
    leg: pl.DataFrame,
    leg_scenario: str,
) -> pl.DataFrame:
    """Express a leg's P&L on the combined portfolio capital base.

    A separately compounded leg is a standalone portfolio. It cannot be added
    to the other leg after a rebalance because both paths compound their own
    capital. This function instead carries each leg's within-period P&L into
    the combined portfolio NAV, making the two contribution paths additive.
    """

    combined_periods = combined.sort("date").with_columns(
        (1.0 + pl.col("gross_return")).cum_prod().alias("local_wealth")
    )
    combined_periods = combined_periods.with_columns(
        pl.col("local_wealth").shift(1).fill_null(1.0).alias("previous_local_wealth")
    )
    # The package returns fixed-notional daily P&L. When the chart compounds
    # the combined return path, each leg's daily P&L must be carried at the
    # combined portfolio's prior wealth for the two contribution paths to add
    # back to that path.
    combined_periods = combined_periods.with_columns(
        pl.col("previous_local_wealth").alias("leg_scale")
    )
    leg_periods = (
        leg.filter(pl.col("scenario") == leg_scenario)
        .sort("date")
        .with_columns((pl.col("portfolio_relative_value") - 1.0).alias("leg_pnl"))
        .join(combined_periods.select("date", "leg_scale"), on="date", how="inner")
        .sort("date")
        .with_columns(
            (1.0 + (pl.col("leg_pnl") * pl.col("leg_scale")).cum_sum()).alias("wealth")
        )
    )
    return leg_periods.select("date", "wealth")


def _regime_series(
    daily: pl.DataFrame,
    scaled_leg_daily: pl.DataFrame,
    scenario_config: ScenarioConfig,
    *,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build comparable portfolio and leg paths for one regime window."""

    window = daily.filter(pl.col("date").is_between(start, end))
    combined_window = window.filter(
        pl.col("scenario") == scenario_config.volatility_scaled_long_short
    )
    strategy = _compound_return_series(
        combined_window,
        "gross_return",
    )
    market = _compound_return_series(
        window.select("date", "market_return").unique("date"),
        "market_return",
    )
    leg_window = scaled_leg_daily.filter(pl.col("date").is_between(start, end))
    long_leg = _compound_leg_contribution(
        combined_window,
        leg_window,
        "scaled_long_leg",
    )
    short_leg = _compound_leg_contribution(
        combined_window,
        leg_window,
        "scaled_short_leg",
    )
    return strategy, market, long_leg, short_leg


def plot_regime_comparison(
    daily: pl.DataFrame,
    scaled_leg_daily: pl.DataFrame,
    path: Path,
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
    mobile_layout: bool = False,
) -> None:
    """Compare the dot-com and AI-led regimes in one 2-by-2 figure."""

    regimes = (
        (date(1998, 10, 8), date(2003, 12, 31), "1998–2003"),
        (date(2025, 4, 3), date(2026, 5, 27), "2025–2026"),
    )
    if mobile_layout:
        figure, flat_axes = plt.subplots(4, 1, figsize=(7.0, 13.0))
        axes = ((flat_axes[0], flat_axes[1]), (flat_axes[2], flat_axes[3]))
    else:
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(12.0, 7.6),
            gridspec_kw={"hspace": 0.22, "wspace": 0.18},
        )

    for row, (start, end, period_label) in enumerate(regimes):
        strategy, market, long_leg, short_leg = _regime_series(
            daily,
            scaled_leg_daily,
            scenario_config,
            start=start,
            end=end,
        )
        combined_axis, legs_axis = axes[row]
        combined_axis.plot(
            strategy.get_column("date").to_list(),
            strategy.get_column("wealth"),
            label="Combined scaled L/S (gross)",
            color=plot_config.volatility_scaled_color,
        )
        combined_axis.plot(
            market.get_column("date").to_list(),
            market.get_column("wealth"),
            label="Russell 1000",
            color=plot_config.muted_text_color,
        )
        legs_axis.plot(
            long_leg.get_column("date").to_list(),
            long_leg.get_column("wealth"),
            label="Low-vol long contribution",
            color=plot_config.low_volatility_color,
        )
        legs_axis.plot(
            short_leg.get_column("date").to_list(),
            short_leg.get_column("wealth"),
            label="High-vol short contribution",
            color=plot_config.high_volatility_color,
        )
        combined_axis.text(
            0.02,
            0.95,
            period_label,
            transform=combined_axis.transAxes,
            color=plot_config.muted_text_color,
            fontsize=10.5,
            va="top",
        )
        for axis in (combined_axis, legs_axis):
            axis.axhline(
                1,
                color=plot_config.zero_line_color,
                linewidth=0.8,
                linestyle=":",
            )
            axis.set_ylabel("Relative wealth")
            _clean_axis(axis, plot_config)
        if row == 0:
            combined_axis.legend(frameon=False, ncol=1 if mobile_layout else 2)
            legs_axis.legend(frameon=False, ncol=1 if mobile_layout else 2)
        else:
            combined_axis.tick_params(axis="x", labelbottom=False)
        legs_axis.set_xlabel("Date")

    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.08,
        top=0.98,
        hspace=0.22,
        wspace=0.18,
    )
    _finish_figure(figure, path, plot_config, tight_layout=False)


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

    for extension in ("png", "svg"):
        renderer(path=figures / f"{stem}.{extension}", mobile_layout=False)
        renderer(path=figures / f"{stem}_mobile.{extension}", mobile_layout=True)


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
        _render_layout_pair(figures, stem, renderer)
