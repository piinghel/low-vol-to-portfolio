"""Transparent weekly target-weight backtest with explicit trading costs."""

from __future__ import annotations

import polars as pl

from .config import BacktestConfig, CostConfig, DataConfig


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
        date_to_signal.join(target_frame, on="signal_date", how="inner")
        .join(
            asset_returns.select(date_col, asset_col, return_col),
            on=[date_col, asset_col],
            how="left",
        )
        .with_columns(
            pl.col(return_col).is_null().alias("missing_return"),
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
            (pl.col("weight") * pl.col("asset_growth")).alias("market_value"),
            (pl.col("weight") * (pl.col("asset_growth") - 1.0)).alias(
                "pnl_since_rebalance"
            ),
        )
    )


def simulate_stock_targets(
    targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    """Simulate fixed quantities between rebalances and expose weight drift."""

    date_col = data_config.date_column
    expanded = _expand_stock_positions(
        targets, asset_returns, date_to_signal, data_config
    )

    daily = (
        expanded.group_by(date_col, "signal_date", "scenario")
        .agg(
            (1.0 + pl.col("pnl_since_rebalance").sum()).alias(
                "portfolio_relative_value"
            ),
            pl.col("market_value").abs().sum().alias("gross_market_value"),
            pl.col("market_value").sum().alias("net_market_value"),
            pl.col("market_value")
            .filter(pl.col("market_value") > 0)
            .sum()
            .alias("long_market_value"),
            (-pl.col("market_value").filter(pl.col("market_value") < 0).sum()).alias(
                "short_market_value"
            ),
            (pl.col("market_value") * pl.col("stock_beta"))
            .sum()
            .alias("beta_market_value"),
            pl.col("missing_return").sum().alias("missing_returns"),
            pl.len().alias("position_return_observations"),
        )
        .sort("scenario", date_col)
        .with_columns(
            pl.col("portfolio_relative_value")
            .shift(1)
            .over(["scenario", "signal_date"], order_by=date_col)
            .fill_null(1.0)
            .alias("previous_relative_value")
        )
        .with_columns(
            (
                pl.col("portfolio_relative_value") / pl.col("previous_relative_value")
                - 1.0
            ).alias("gross_return"),
            (pl.col("gross_market_value") / pl.col("portfolio_relative_value")).alias(
                "gross_exposure"
            ),
            (pl.col("net_market_value") / pl.col("portfolio_relative_value")).alias(
                "net_exposure"
            ),
            (pl.col("long_market_value") / pl.col("portfolio_relative_value")).alias(
                "long_exposure"
            ),
            (pl.col("short_market_value") / pl.col("portfolio_relative_value")).alias(
                "short_exposure"
            ),
            (pl.col("beta_market_value") / pl.col("portfolio_relative_value")).alias(
                "stock_beta"
            ),
        )
        .drop("previous_relative_value")
        .join(date_to_signal.select(date_col, "market_return"), on=date_col, how="left")
    )
    return daily


def compute_realized_turnover(
    targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    """Compare new targets with drifted holdings at each execution close."""

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
        .with_columns(
            (
                1.0
                + pl.col("pnl_since_rebalance").sum().over(["scenario", "signal_date"])
            ).alias("ending_portfolio_value")
        )
        .with_columns(
            (pl.col("market_value") / pl.col("ending_portfolio_value")).alias(
                "old_weight"
            )
        )
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
        .with_columns(
            (pl.col("gross_return") - pl.col("equity_cost")).alias("net_return")
        )
        .sort("scenario", "date")
    )
