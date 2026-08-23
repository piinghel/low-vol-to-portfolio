"""Performance and difficult-regime figures for the low-volatility article."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, NullLocator

from .config import PlotConfig, ScenarioConfig
from .frame_validation import require_finite_float
from .metrics import cumulative_returns, drawdown_series
from .plot_style import clean_axis, finish_figure


def _comparison_style(
    scenario_config: ScenarioConfig,
    plot_config: PlotConfig,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return shared labels and colors for the two implementations."""

    scenarios = [
        scenario_config.naive_equal_weight_long_short,
        scenario_config.volatility_scaled_long_short,
    ]
    labels = {
        scenario_config.naive_equal_weight_long_short: "Equal-weight\nlong/short",
        scenario_config.volatility_scaled_long_short: "Volatility-scaled\nlong/short",
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
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(5.2, 7.4) if mobile_layout else (9.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.08},
    )
    wealth_axis, drawdown_axis = axes
    wealth_endpoints: dict[str, tuple[date, float]] = {}
    for scenario in scenarios:
        wealth = performance.filter(pl.col("scenario") == scenario).sort("date")
        wealth_dates = wealth.get_column("date").to_list()
        wealth_values = wealth.get_column("wealth")
        wealth_axis.plot(wealth_dates, wealth_values, color=colors[scenario])
        wealth_endpoints[scenario] = (
            wealth_dates[-1],
            require_finite_float(wealth_values[-1], f"final wealth for {scenario}"),
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
    wealth_axis.set_ylabel("Cumulative return")
    last_date = max(date_value for date_value, _ in wealth_endpoints.values())
    label_date = last_date + timedelta(days=80)
    wealth_axis.set_xlim(
        performance.get_column("date").min(),
        last_date + timedelta(days=180),
    )
    for index, scenario in enumerate(
        sorted(scenarios, key=lambda item: wealth_endpoints[item][1])
    ):
        _, value = wealth_endpoints[scenario]
        wealth_axis.annotate(
            labels[scenario],
            xy=(label_date, value),
            xytext=(0, -10 if index == 0 else 10),
            textcoords="offset points",
            ha="left",
            va="center",
            color=colors[scenario],
            fontsize=10.5,
            fontweight="normal",
        )
    drawdown_axis.set_ylabel("Drawdown (%)")
    for axis in axes:
        clean_axis(axis, plot_config)
    wealth_axis.grid(False, axis="y")
    for tick in wealth_axis.get_yticks():
        if abs(tick - 1.0) > 1e-9:
            wealth_axis.axhline(
                tick,
                color=plot_config.grid_color,
                linewidth=0.8,
                zorder=0,
            )
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.09,
        top=0.98,
        hspace=0.08,
    )
    finish_figure(figure, path, plot_config, tight_layout=False)


def _compound_return_series(
    frame: pl.DataFrame,
    return_column: str,
    *,
    baseline_at_first_date: bool = False,
) -> pl.DataFrame:
    ordered = frame.sort("date").with_row_index("row_number")
    period_return = pl.col(return_column).fill_null(0.0)
    if baseline_at_first_date:
        period_return = (
            pl.when(pl.col("row_number") == 0).then(0.0).otherwise(period_return)
        )
    return ordered.select(
        "date",
        (1.0 + period_return).cum_prod().alias("wealth"),
    )


def _compound_leg_contribution(
    strategy: pl.DataFrame,
    leg: pl.DataFrame,
    leg_scenario: str,
    *,
    baseline_at_first_date: bool = False,
) -> pl.DataFrame:
    """Carry a basket's P&L on the strategy capital base."""

    strategy_periods = strategy.sort("date").with_row_index("row_number")
    strategy_return = pl.col("gross_return").fill_null(0.0)
    if baseline_at_first_date:
        strategy_return = (
            pl.when(pl.col("row_number") == 0).then(0.0).otherwise(strategy_return)
        )
    strategy_periods = strategy_periods.with_columns(
        (1.0 + strategy_return).cum_prod().alias("local_wealth")
    ).with_columns(pl.col("local_wealth").shift(1).fill_null(1.0).alias("leg_scale"))
    leg_periods = (
        leg.filter(pl.col("scenario") == leg_scenario)
        .sort("date")
        .with_columns((pl.col("portfolio_relative_value") - 1.0).alias("leg_pnl"))
        .join(strategy_periods.select("date", "leg_scale"), on="date", how="inner")
        .sort("date")
        .with_row_index("row_number")
        .with_columns(
            pl.when(pl.lit(baseline_at_first_date) & (pl.col("row_number") == 0))
            .then(0.0)
            .otherwise(pl.col("leg_pnl"))
            .alias("leg_pnl")
        )
        .with_columns(
            (1.0 + (pl.col("leg_pnl") * pl.col("leg_scale")).cum_sum()).alias("wealth")
        )
    )
    return leg_periods.select("date", "wealth")


def _episode_series(
    daily: pl.DataFrame,
    scaled_leg_daily: pl.DataFrame,
    scenario_config: ScenarioConfig,
    *,
    start: date,
    end: date,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build comparable portfolio, market, and leg paths for one episode."""

    window = daily.filter(pl.col("date").is_between(start, end))
    strategy_window = window.filter(
        pl.col("scenario") == scenario_config.volatility_scaled_long_short
    )
    strategy = _compound_return_series(
        strategy_window,
        "gross_return",
        baseline_at_first_date=True,
    )
    market = _compound_return_series(
        window.select("date", "market_return").unique("date"),
        "market_return",
        baseline_at_first_date=True,
    )
    leg_window = scaled_leg_daily.filter(pl.col("date").is_between(start, end))
    long_leg = _compound_leg_contribution(
        strategy_window,
        leg_window,
        "scaled_long_leg",
        baseline_at_first_date=True,
    )
    short_leg = _compound_leg_contribution(
        strategy_window,
        leg_window,
        "scaled_short_leg",
        baseline_at_first_date=True,
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
    """Show the dot-com rally, reversal, and the portfolio's two legs."""

    start = date(1998, 10, 8)
    portfolio_trough = date(2000, 3, 9)
    end = date(2001, 4, 3)
    strategy, market, long_leg, short_leg = _episode_series(
        daily,
        scaled_leg_daily,
        scenario_config,
        start=start,
        end=end,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(5.2, 7.4) if mobile_layout else (9.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.12},
    )
    wealth_axis, legs_axis = axes
    wealth_axis.plot(
        market.get_column("date").to_list(),
        market.get_column("wealth"),
        label="Russell 1000",
        color=plot_config.naive_long_short_color,
        linewidth=1.5,
    )
    wealth_axis.plot(
        strategy.get_column("date").to_list(),
        strategy.get_column("wealth"),
        label="Low-vol portfolio",
        color=plot_config.volatility_scaled_color,
        linewidth=1.7,
    )
    legs_axis.plot(
        long_leg.get_column("date").to_list(),
        (long_leg.get_column("wealth") - 1.0) * 100.0,
        label="Long book",
        color=plot_config.low_volatility_color,
        linewidth=1.5,
    )
    legs_axis.plot(
        short_leg.get_column("date").to_list(),
        (short_leg.get_column("wealth") - 1.0) * 100.0,
        label="Short book",
        color=plot_config.high_volatility_color,
        linewidth=1.5,
    )

    wealth_axis.set_ylabel("Cumulative wealth")
    wealth_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    wealth_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".") + "×")
    )
    legs_axis.set_ylabel("Contribution (pp)")
    legs_axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    legs_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}"))
    legs_axis.axhline(0, color=plot_config.zero_line_color, linewidth=0.9)

    for axis in axes:
        axis.set_xlim(start, end)
        axis.axvspan(start, portfolio_trough, color=plot_config.grid_color, alpha=0.18)
        axis.axvline(
            portfolio_trough,
            color=plot_config.zero_line_color,
            linewidth=0.9,
        )
        clean_axis(axis, plot_config)
    wealth_axis.text(
        portfolio_trough,
        0.97,
        "9 Mar 2000\nportfolio trough",
        transform=wealth_axis.get_xaxis_transform(),
        ha="right",
        va="top",
        color=plot_config.muted_text_color,
        fontsize=10.0,
    )
    wealth_axis.text(
        date(1999, 5, 1),
        0.06,
        "Market rally\nfactor drawdown",
        transform=wealth_axis.get_xaxis_transform(),
        color=plot_config.muted_text_color,
        fontsize=10.0,
    )
    wealth_axis.text(
        date(2000, 8, 15),
        0.06,
        "Factor recovery",
        transform=wealth_axis.get_xaxis_transform(),
        color=plot_config.muted_text_color,
        fontsize=10.0,
    )
    legs_axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    legs_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    handles = []
    labels = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2 if mobile_layout else 4,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.1,
        fontsize=10.0 if mobile_layout else 10.5,
    )
    for label in legend.get_texts():
        label.set_color(plot_config.text_color)
    figure.subplots_adjust(
        left=0.19 if mobile_layout else 0.11,
        right=0.98,
        bottom=0.10,
        top=0.88,
        hspace=0.12,
    )
    finish_figure(
        figure,
        path,
        plot_config,
        tight_layout=False,
        tick_label_size=10.4 if not mobile_layout else 10.0,
        axis_label_size=11.5 if not mobile_layout else 11.0,
        legend_size=10.5 if not mobile_layout else 10.0,
    )
