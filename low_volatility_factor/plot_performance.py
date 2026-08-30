"""Performance and difficult-regime figures for the low-volatility article."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from math import ceil, floor
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


def _panel_title_color(plot_config: PlotConfig) -> str:
    """Return the shared high-contrast color for performance panel titles."""

    if plot_config.background_color.lower() == "#ffffff":
        return "#000000"
    return plot_config.text_color


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
) -> None:
    """Plot cumulative wealth and drawdowns in one aligned figure."""

    scenarios, labels, colors = _comparison_style(scenario_config, plot_config)
    filtered = daily.filter(pl.col("scenario").is_in(scenarios))
    performance = cumulative_returns(filtered, return_column="net_return")
    drawdowns = drawdown_series(filtered, return_column="net_return")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.2),
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
    panel_title_color = _panel_title_color(plot_config)
    wealth_axis.set_title(
        "Growth of $1 (log scale)",
        loc="left",
        pad=9,
        color=panel_title_color,
        fontweight="bold",
    )
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
    drawdown_axis.set_title(
        "Drawdown (%)",
        loc="left",
        pad=9,
        color=panel_title_color,
        fontweight="bold",
    )
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
) -> None:
    """Contrast a completed rally/reversal with the still-open recent rally."""

    episodes: tuple[tuple[date, date, date | None, str, int], ...] = (
        (
            date(1998, 10, 8),
            date(2001, 4, 3),
            date(2000, 3, 9),
            "A  Dot-com: rally, then reversal",
            6,
        ),
        (
            date(2025, 4, 3),
            date(2026, 5, 27),
            None,
            "B  AI rally",
            3,
        ),
    )
    episode_series = [
        (
            (start, end, turn, title, tick_months),
            _episode_series(
                daily,
                scaled_leg_daily,
                scenario_config,
                start=start,
                end=end,
            ),
        )
        for start, end, turn, title, tick_months in episodes
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 7.8),
        gridspec_kw={"height_ratios": [1.12, 1.0]},
    )
    episode_axes = (
        (axes[0, 0], axes[1, 0]),
        (axes[0, 1], axes[1, 1]),
    )
    panel_title_color = _panel_title_color(plot_config)

    for (episode, series), (wealth_axis, legs_axis) in zip(
        episode_series, episode_axes, strict=True
    ):
        start, end, turn, title, tick_months = episode
        strategy, market, long_leg, short_leg = series
        market_dates = market.get_column("date").to_list()
        market_wealth = market.get_column("wealth").to_list()
        strategy_dates = strategy.get_column("date").to_list()
        strategy_wealth = strategy.get_column("wealth").to_list()
        long_dates = long_leg.get_column("date").to_list()
        long_contribution = ((long_leg.get_column("wealth") - 1.0) * 100.0).to_list()
        short_dates = short_leg.get_column("date").to_list()
        short_contribution = ((short_leg.get_column("wealth") - 1.0) * 100.0).to_list()
        wealth_axis.plot(
            market_dates,
            market_wealth,
            color=plot_config.naive_long_short_color,
            linewidth=1.5,
        )
        wealth_axis.plot(
            strategy_dates,
            strategy_wealth,
            color=plot_config.volatility_scaled_color,
            linewidth=1.7,
        )
        legs_axis.plot(
            long_dates,
            long_contribution,
            color=plot_config.low_volatility_color,
            linewidth=1.5,
        )
        legs_axis.plot(
            short_dates,
            short_contribution,
            color=plot_config.high_volatility_color,
            linewidth=1.5,
        )

        label_size = 9.4
        label_date = end + (end - start) * 0.025
        for axis, values, label, color, offset in (
            (
                wealth_axis,
                market_wealth,
                "Russell 1000",
                plot_config.muted_text_color,
                4,
            ),
            (
                wealth_axis,
                strategy_wealth,
                "Low-vol",
                plot_config.volatility_scaled_color,
                -4,
            ),
            (
                legs_axis,
                long_contribution,
                "Long low-vol",
                plot_config.low_volatility_color,
                4,
            ),
            (
                legs_axis,
                short_contribution,
                "Short high-vol",
                plot_config.high_volatility_color,
                -4,
            ),
        ):
            axis.annotate(
                label,
                (label_date, values[-1]),
                xytext=(0, offset),
                textcoords="offset points",
                ha="left",
                va="center",
                color=color,
                fontsize=label_size,
                fontweight="bold",
            )

        wealth_axis.set_title(
            f"{title}\nGrowth of $1",
            loc="left",
            pad=10,
            color=panel_title_color,
            fontsize=12.0,
            fontweight="bold",
        )
        for axis in (wealth_axis, legs_axis):
            tick_locator = mdates.MonthLocator(interval=tick_months)
            axis.set_xticks(
                tick_locator.tick_values(
                    datetime.combine(start, time.min),
                    datetime.combine(end, time.min),
                )
            )
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            axis.set_xlim(start, end + (end - start) * 0.18)
            clean_axis(axis, plot_config)
        wealth_axis.tick_params(axis="x", labelbottom=False)
        episode_wealth = market_wealth + strategy_wealth + [1.0]
        wealth_axis.set_ylim(
            floor((min(episode_wealth) - 0.02) / 0.05) * 0.05,
            ceil((max(episode_wealth) + 0.02) / 0.05) * 0.05,
        )
        wealth_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        wealth_axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".") + "×")
        )
        episode_contributions = long_contribution + short_contribution + [0.0]
        legs_axis.set_ylim(
            floor((min(episode_contributions) - 1.0) / 5.0) * 5.0,
            ceil((max(episode_contributions) + 1.0) / 5.0) * 5.0,
        )
        legs_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        legs_axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:+.0f} pp")
        )
        legs_axis.set_title(
            "Long/short contribution (pp)",
            loc="left",
            pad=8,
            color=panel_title_color,
            fontsize=10.5,
            fontweight="normal",
        )
        if turn is not None:
            for axis in (wealth_axis, legs_axis):
                axis.axvline(
                    turn,
                    color=plot_config.zero_line_color,
                    linewidth=0.9,
                    linestyle=(0, (2, 3)),
                )
    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.07,
        top=0.96,
        hspace=0.32,
        wspace=0.20,
    )
    finish_figure(
        figure,
        path,
        plot_config,
        tight_layout=False,
        tick_label_size=10.4,
        axis_label_size=11.5,
        legend_size=10.5,
    )
