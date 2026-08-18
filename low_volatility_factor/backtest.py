"""Transparent floating-weight backtest with explicit trading costs."""

from __future__ import annotations

import polars as pl
from portfolio_management.analysis import forward_fill_prices
from tranching_pnl.pnl import PnLConfig, compute_pnl_results

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
    prepared = held.filter(
        pl.col("last_covered_date").is_not_null()
        & (pl.col(date_col) >= pl.col("first_covered_date"))
        & (pl.col(date_col) <= pl.col("last_covered_date"))
    ).drop("last_covered_date", "first_covered_date")
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
        pl.col(return_col).is_null().alias("missing_return"),
    ).with_columns(pl.col(return_col).fill_null(0.0))


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


def _price_panel(
    price_data: pl.DataFrame,
    calendar: pl.DataFrame,
    data_config: DataConfig,
) -> pl.DataFrame:
    """Forward-fill adjusted prices through the shared package boundary."""

    return forward_fill_prices(
        price_data=price_data.select(
            data_config.date_column,
            data_config.asset_column,
            data_config.adjusted_price_column,
        ),
        calendar=calendar.select(data_config.date_column),
        assets=price_data.select(data_config.asset_column).unique(),
        date_column=data_config.date_column,
        asset_id_column=data_config.asset_column,
        price_column=data_config.adjusted_price_column,
    )


def _build_package_book(
    targets: pl.DataFrame,
    schedule: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    price_data: pl.DataFrame,
    data_config: DataConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Materialize effective quantities and execution notionals for package PnL."""

    date_col = data_config.date_column
    asset_col = data_config.asset_column
    calendar = date_to_signal.select(date_col).unique().sort(date_col)
    price_calendar = (
        pl.concat(
            [
                calendar,
                schedule.select(pl.col("execution_date").alias(date_col)),
            ],
            how="vertical",
        )
        .unique()
        .sort(date_col)
    )
    prices = _price_panel(price_data, price_calendar, data_config)
    # Attach the execution price to each target row. The package uses the
    # resulting quantity and adjusted price differences for PnL.
    target_effective = (
        schedule.select("signal_date", "execution_date", "effective_return_date")
        .join(targets, on="signal_date")
        .join(
            prices.rename({date_col: "execution_date"}),
            on=["execution_date", asset_col],
            how="left",
        )
        .rename({"px_last": "execution_price"})
        .with_columns(
            (pl.col("weight") / pl.col("execution_price")).alias("final_target_qty")
        )
    )
    if target_effective.get_column("execution_price").null_count():
        raise ValueError("Missing execution prices for target positions")

    book = (
        date_to_signal.select(date_col, "signal_date")
        .join(
            target_effective.select(
                "signal_date", asset_col, "scenario", "final_target_qty"
            ),
            on="signal_date",
            how="inner",
        )
        .select(date_col, asset_col, "scenario", "final_target_qty")
        .unique([date_col, asset_col, "scenario"])
    )

    signal_sequence = (
        schedule.select("signal_date", "effective_return_date")
        .unique()
        .sort("signal_date")
        .with_columns(pl.col("signal_date").shift(1).alias("previous_signal_date"))
    )
    current = target_effective.select(
        "signal_date",
        "effective_return_date",
        "scenario",
        asset_col,
        "execution_price",
        pl.col("weight").alias("new_value"),
    )
    previous = (
        signal_sequence.join(
            target_effective.select(
                "signal_date", "scenario", asset_col, "final_target_qty"
            ),
            left_on="previous_signal_date",
            right_on="signal_date",
            how="inner",
        )
        .join(
            current.select("signal_date", asset_col, "execution_price"),
            on=["signal_date", asset_col],
            how="left",
        )
        .select(
            "signal_date",
            "effective_return_date",
            "scenario",
            asset_col,
            (pl.col("final_target_qty") * pl.col("execution_price")).alias("old_value"),
        )
    )
    executions = (
        current.join(
            previous,
            on=["signal_date", "effective_return_date", "scenario", asset_col],
            how="full",
            coalesce=True,
        )
        .with_columns(
            pl.col("new_value").fill_null(0.0),
            pl.col("old_value").fill_null(0.0),
        )
        .with_columns(
            (pl.col("new_value") - pl.col("old_value")).alias("final_execution_value")
        )
        .filter(pl.col("final_execution_value") != 0)
        .select(
            pl.col("effective_return_date").alias(date_col),
            asset_col,
            "scenario",
            "final_execution_value",
        )
    )
    metadata = (
        date_to_signal.select(date_col, "signal_date")
        .join(
            targets.select("signal_date", asset_col, "scenario", "stock_beta"),
            on="signal_date",
            how="inner",
        )
        .select(date_col, asset_col, "scenario", "stock_beta")
    )
    return book, executions, metadata


def simulate_stock_targets(
    targets: pl.DataFrame,
    price_data: pl.DataFrame,
    date_to_signal: pl.DataFrame,
    schedule: pl.DataFrame,
    data_config: DataConfig,
    cost_config: CostConfig,
) -> pl.DataFrame:
    """Compute quantity-based PnL, costs, and floating exposures via packages."""

    date_col = data_config.date_column
    book, executions, metadata = _build_package_book(
        targets, schedule, date_to_signal, price_data, data_config
    )
    portfolio_names = targets.get_column("scenario").unique().sort().to_list()
    pnl = compute_pnl_results(
        df_book=book,
        df_executions=executions,
        df_price=price_data.select(
            date_col, data_config.asset_column, data_config.adjusted_price_column
        ).rename({data_config.adjusted_price_column: "px_last"}),
        reference_calendar=pl.concat(
            [
                date_to_signal.select(date_col),
                schedule.select(pl.col("execution_date").alias(date_col)),
            ],
            how="vertical",
        )
        .unique()
        .sort(date_col),
        portfolios=portfolio_names,
        pnl_config=PnLConfig(
            asset_id_column=data_config.asset_column,
            price_column="px_last",
            date_column=date_col,
            quantity_column="final_target_qty",
            portfolio_column="scenario",
            notional_mode="fixed",
        ),
    )
    raw = pnl.get_raw_pnl()
    after_costs = pnl.get_portfolio_pnl(
        amount_invested=1.0,
        transaction_costs_mapping={
            name: cost_config.equity_cost_bps / 10_000 for name in portfolio_names
        },
        to_return=False,
    )
    turnover = pnl.compute_turnover(amount_invested=1.0, output_format="long")
    if "portfolio_bucket" in turnover.columns:
        turnover = turnover.rename({"portfolio_bucket": "scenario"})
    position_prices = _price_panel(
        price_data,
        pl.concat(
            [
                book.select(date_col),
                schedule.select(pl.col("execution_date").alias(date_col)),
            ],
            how="vertical",
        )
        .unique()
        .sort(date_col),
        data_config,
    )
    positions = pnl.get_position_floating_weights(
        price_data=position_prices.rename(
            {data_config.adjusted_price_column: "px_last"}
        ),
        amount_invested=1.0,
        allow_missing_prices=False,
    )
    exposures = (
        positions.join(metadata, on=[date_col, data_config.asset_column, "scenario"])
        .group_by(date_col, "scenario")
        .agg(
            pl.col("floating_weight").abs().sum().alias("gross_exposure"),
            pl.col("floating_weight").sum().alias("net_exposure"),
            pl.col("floating_weight")
            .filter(pl.col("floating_weight") > 0)
            .sum()
            .alias("long_exposure"),
            (
                -pl.col("floating_weight").filter(pl.col("floating_weight") < 0).sum()
            ).alias("short_exposure"),
            (pl.col("floating_weight") * pl.col("stock_beta"))
            .sum()
            .alias("stock_beta"),
        )
    )
    daily = (
        raw.rename({name: f"gross_pnl_{name}" for name in portfolio_names})
        .join(
            after_costs.rename({name: f"net_pnl_{name}" for name in portfolio_names}),
            on=date_col,
        )
        .unpivot(index=date_col, variable_name="value_type", value_name="value")
        .with_columns(
            pl.col("value_type")
            .str.extract(r"^(gross_pnl|net_pnl)_(.*)$", 1)
            .alias("fee_state"),
            pl.col("value_type")
            .str.extract(r"^(?:gross_pnl|net_pnl)_(.*)$", 1)
            .alias("scenario"),
        )
        .pivot(index=[date_col, "scenario"], on="fee_state", values="value")
        .rename({"gross_pnl": "gross_pnl", "net_pnl": "net_pnl"})
        .join(exposures, on=[date_col, "scenario"], how="left")
        .join(turnover, on=[date_col, "scenario"], how="left")
        .join(
            date_to_signal.select(date_col, "signal_date", "market_return"), on=date_col
        )
        .with_columns(
            pl.col("turnover").fill_null(0.0).alias("equity_turnover"),
            (pl.col("gross_pnl") - pl.col("net_pnl")).alias("equity_cost"),
            pl.col("gross_pnl").alias("gross_return"),
            pl.col("net_pnl").alias("net_return"),
            (1.0 + pl.col("gross_pnl")).alias("portfolio_relative_value"),
        )
        .select(
            date_col,
            "signal_date",
            "scenario",
            "portfolio_relative_value",
            "gross_pnl",
            "net_pnl",
            "gross_return",
            "net_return",
            "gross_exposure",
            "net_exposure",
            "long_exposure",
            "short_exposure",
            "stock_beta",
            "equity_turnover",
            "equity_cost",
            "market_return",
        )
        .sort("scenario", date_col)
    )
    return daily
