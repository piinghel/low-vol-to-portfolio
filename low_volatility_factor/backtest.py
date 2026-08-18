"""Transparent floating-weight backtest with explicit trading costs."""

from __future__ import annotations

import polars as pl

from .config import BacktestConfig, CostConfig, DataConfig


def prepare_held_returns(
    targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    """Join returns to held positions and close trailing uncovered rows.

    The source panel is sparse for some older securities. Missing dates inside
    a security's observed life therefore carry a zero return; dates after its
    last observation close the position, matching the package's delisting
    convention.
    """

    date_col = data_config.date_column
    asset_col = data_config.asset_column
    return_col = data_config.total_return_column
    held = (
        date_to_signal.select(date_col, "signal_date")
        .join(targets, on="signal_date")
        .join(
            asset_returns.select(date_col, asset_col, return_col),
            on=[date_col, asset_col],
            how="left",
        )
        .sort("scenario", "signal_date", asset_col, date_col)
        .with_columns(
            pl.when(pl.col(return_col).is_not_null())
            .then(pl.col(date_col))
            .otherwise(None)
            .max()
            .over(["scenario", "signal_date", asset_col])
            .alias("last_covered_date")
        )
        .with_columns(
            pl.when(pl.col(return_col).is_not_null())
            .then(pl.col(date_col))
            .otherwise(None)
            .min()
            .over(["scenario", "signal_date", asset_col])
            .alias("first_covered_date")
        )
    )
    prepared = (
        held.filter(
            pl.col("last_covered_date").is_not_null()
            & (pl.col(date_col) >= pl.col("first_covered_date"))
            & (pl.col(date_col) <= pl.col("last_covered_date"))
        )
        .drop("last_covered_date", "first_covered_date")
        .with_columns(pl.col(return_col).fill_null(0.0))
    )
    invalid_returns = prepared.filter(pl.col(return_col) <= -1.0)
    if not invalid_returns.is_empty():
        offenders = invalid_returns.select(
            date_col, "signal_date", "scenario", asset_col, return_col
        ).head(10)
        raise ValueError(
            "Held positions contain returns at or below -100%; "
            f"first offenders: {offenders.to_dicts()}"
        )
    return prepared.with_columns(
        pl.lit(False).alias("missing_return"),
    )


def build_execution_schedule(
    signal_dates: pl.DataFrame,
    market_calendar: pl.DataFrame,
    data_config: DataConfig,
    backtest_config: BacktestConfig,
) -> pl.DataFrame:
    """Map signal close to execution close and first subsequent return date."""

    date_col = data_config.date_column
    delay = backtest_config.execution_delay_trading_days
    calendar = (
        market_calendar.select(date_col)
        .unique()
        .sort(date_col)
        .with_columns(
            pl.col(date_col).shift(-delay).alias("execution_date"),
            pl.col(date_col).shift(-(delay + 1)).alias("effective_return_date"),
        )
        .rename({date_col: "signal_date"})
    )
    return (
        signal_dates.rename({date_col: "signal_date"})
        .join(calendar, on="signal_date", how="inner")
        .drop_nulls(["execution_date", "effective_return_date"])
        .sort("signal_date")
    )


def map_dates_to_signal_periods(
    schedule: pl.DataFrame,
    market_returns: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    date_col = data_config.date_column
    return (
        market_returns.select(date_col, "market_return")
        .sort(date_col)
        .join_asof(
            schedule.select("signal_date", "effective_return_date").sort(
                "effective_return_date"
            ),
            left_on=date_col,
            right_on="effective_return_date",
            strategy="backward",
        )
        .drop_nulls("signal_date")
    )


def _expand_stock_positions(
    targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    return_col = data_config.total_return_column
    target_frame = targets
    if "stock_beta" not in target_frame.columns:
        target_frame = target_frame.with_columns(pl.lit(0.0).alias("stock_beta"))

    return (
        prepare_held_returns(target_frame, asset_returns, date_to_signal, data_config)
        .with_columns(
            pl.col(return_col).fill_null(0.0),
        )
        .sort("scenario", "signal_date", asset_col, date_col)
        .with_columns(
            (1.0 + pl.col(return_col))
            .cum_prod()
            .over(["scenario", "signal_date", asset_col], order_by=date_col)
            .alias("asset_growth")
        )
        .with_columns(
            (pl.col("weight") * pl.col("asset_growth")).alias("floating_weight")
        )
        .with_columns(
            (pl.col("floating_weight") / (1.0 + pl.col(return_col))).alias(
                "start_of_day_weight"
            )
        )
        .with_columns(
            (pl.col("start_of_day_weight") * pl.col(return_col)).alias("daily_pnl")
        )
    )


def simulate_stock_targets(
    targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    """Simulate floating weights between rebalances and expose their drift."""

    date_col = data_config.date_column
    expanded = _expand_stock_positions(
        targets, asset_returns, date_to_signal, data_config
    )

    daily = (
        expanded.group_by(date_col, "signal_date", "scenario")
        .agg(
            (1.0 + pl.col("daily_pnl").sum()).alias("portfolio_relative_value"),
            pl.col("daily_pnl").sum().alias("gross_pnl"),
            pl.col("floating_weight").abs().sum().alias("gross_market_value"),
            pl.col("floating_weight").sum().alias("net_market_value"),
            pl.col("floating_weight")
            .filter(pl.col("floating_weight") > 0)
            .sum()
            .alias("long_market_value"),
            (
                -pl.col("floating_weight").filter(pl.col("floating_weight") < 0).sum()
            ).alias("short_market_value"),
            (pl.col("floating_weight") * pl.col("stock_beta"))
            .sum()
            .alias("beta_market_value"),
            pl.col("missing_return").sum().alias("missing_returns"),
            pl.len().alias("position_return_observations"),
        )
        .sort("scenario", date_col)
        .with_columns(
            pl.col("gross_pnl")
            .cum_sum()
            .over("scenario", order_by=date_col)
            .alias("cumulative_gross_pnl")
        )
        .with_columns(
            (
                1.0
                + pl.col("cumulative_gross_pnl")
                .shift(1)
                .over("scenario")
                .fill_null(0.0)
            ).alias("previous_gross_nav")
        )
        .with_columns(
            (pl.col("gross_pnl") / pl.col("previous_gross_nav")).alias("gross_return"),
            pl.col("gross_market_value").alias("gross_exposure"),
            pl.col("net_market_value").alias("net_exposure"),
            pl.col("long_market_value").alias("long_exposure"),
            pl.col("short_market_value").alias("short_exposure"),
            pl.col("beta_market_value").alias("stock_beta"),
        )
        .drop("cumulative_gross_pnl")
        .join(date_to_signal.select(date_col, "market_return"), on=date_col, how="left")
    )
    return daily


def compute_realized_turnover(
    targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    """Compare new targets with outgoing floating weights at each rebalance."""

    asset_col = data_config.asset_column
    date_col = data_config.date_column
    expanded = _expand_stock_positions(
        targets, asset_returns, date_to_signal, data_config
    )
    period_ends = expanded.group_by("scenario", "signal_date").agg(
        pl.col(date_col).max().alias("period_end_date")
    )
    ending_holdings = (
        expanded.join(period_ends, on=["scenario", "signal_date"], how="inner")
        .filter(pl.col(date_col) == pl.col("period_end_date"))
        .with_columns(pl.col("floating_weight").alias("old_weight"))
    )
    next_signals = (
        targets.select("scenario", "signal_date")
        .unique()
        .sort("scenario", "signal_date")
        .with_columns(
            pl.col("signal_date")
            .shift(-1)
            .over("scenario", order_by="signal_date")
            .alias("next_signal_date")
        )
    )
    old_at_next_trade = (
        ending_holdings.join(next_signals, on=["scenario", "signal_date"], how="left")
        .drop_nulls("next_signal_date")
        .select(
            "scenario",
            pl.col("next_signal_date").alias("signal_date"),
            asset_col,
            "old_weight",
        )
    )
    new_targets = targets.select(
        "scenario", "signal_date", asset_col, pl.col("weight").alias("new_weight")
    )
    return (
        new_targets.join(
            old_at_next_trade,
            on=["scenario", "signal_date", asset_col],
            how="full",
            coalesce=True,
        )
        .with_columns(
            pl.col("new_weight").fill_null(0.0),
            pl.col("old_weight").fill_null(0.0),
        )
        .group_by("signal_date", "scenario")
        .agg(
            (pl.col("new_weight") - pl.col("old_weight"))
            .abs()
            .sum()
            .alias("equity_turnover")
        )
        .sort("signal_date", "scenario")
    )


def apply_transaction_costs(
    daily: pl.DataFrame,
    target_turnover: pl.DataFrame,
    schedule: pl.DataFrame,
    cost_config: CostConfig,
) -> pl.DataFrame:
    equity_costs = (
        target_turnover.join(
            schedule.select("signal_date", "effective_return_date"),
            on="signal_date",
            how="left",
        )
        .rename({"effective_return_date": "date"})
        .with_columns(
            (pl.col("equity_turnover") * cost_config.equity_cost_bps / 10_000).alias(
                "equity_cost"
            )
        )
    )
    return (
        daily.join(
            equity_costs.select("date", "scenario", "equity_turnover", "equity_cost"),
            on=["date", "scenario"],
            how="left",
        )
        .with_columns(
            pl.col("equity_turnover").fill_null(0.0),
            pl.col("equity_cost").fill_null(0.0),
        )
        .with_columns((pl.col("gross_pnl") - pl.col("equity_cost")).alias("net_pnl"))
        .with_columns(
            pl.col("net_pnl")
            .cum_sum()
            .over("scenario", order_by="date")
            .alias("cumulative_net_pnl")
        )
        .with_columns(
            (
                pl.col("net_pnl")
                / (
                    1.0
                    + pl.col("cumulative_net_pnl")
                    .shift(1)
                    .over("scenario")
                    .fill_null(0.0)
                )
            ).alias("net_return")
        )
        .drop("cumulative_net_pnl")
        .sort("scenario", "date")
    )
