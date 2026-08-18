"""Portfolio selection and target-weight construction."""

from __future__ import annotations

import polars as pl

from .config import (
    BacktestConfig,
    BucketConfig,
    DataConfig,
    ScenarioConfig,
    SignalConfig,
    SizingConfig,
)
from .frames import Frame, as_lazy, require_columns


def select_rebalance_signal_dates(
    dates: Frame,
    data_config: DataConfig,
    backtest_config: BacktestConfig,
) -> pl.DataFrame:
    """Select every Nth trading week on or after the requested weekday."""

    date_col = data_config.date_column
    frame = as_lazy(dates)
    require_columns(frame, [date_col], "dates")
    candidate_dates = (
        frame.select(date_col)
        .unique()
        .with_columns(
            pl.col(date_col).dt.truncate("1w").alias("week_start"),
            pl.col(date_col).dt.weekday().alias("weekday"),
        )
        .filter(pl.col("weekday") >= backtest_config.rebalance_weekday)
        .group_by("week_start")
        .agg(pl.col(date_col).min())
        .sort(date_col)
        .select(date_col)
        .collect()
    )
    return (
        candidate_dates.with_row_index("week_index")
        .filter(pl.col("week_index") % backtest_config.rebalance_interval_weeks == 0)
        .select(date_col)
    )


def _equal_weight_scenario(
    candidates: pl.LazyFrame,
    *,
    scenario: str,
    signed: bool,
) -> pl.LazyFrame:
    group_columns = ["signal_date", "leg"]
    sign = (
        pl.when(pl.col("leg") == "long").then(1.0).otherwise(-1.0)
        if signed
        else pl.lit(1.0)
    )
    return (
        candidates.with_columns(pl.len().over(group_columns).alias("leg_size"))
        .with_columns((sign / pl.col("leg_size")).alias("weight"))
        .with_columns(pl.lit(scenario).alias("scenario"))
    )


def build_stage_targets(
    signal_snapshots: Frame,
    data_config: DataConfig,
    signal_config: SignalConfig,
    bucket_config: BucketConfig,
    sizing_config: SizingConfig,
    scenario_config: ScenarioConfig,
) -> pl.DataFrame:
    """Build the long-only, naive L/S, and stock-volatility-scaled targets."""

    frame = as_lazy(signal_snapshots)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    bucket_col = bucket_config.bucket_column
    sizing_col = signal_config.sizing_volatility_column
    required = [date_col, asset_col, bucket_col, sizing_col, "stock_beta"]
    require_columns(frame, required, "signal_snapshots")

    candidates = (
        frame.filter(
            pl.col(bucket_col).is_in(
                [
                    bucket_config.low_volatility_bucket,
                    bucket_config.high_volatility_bucket,
                ]
            )
        )
        .drop_nulls([sizing_col, "stock_beta"])
        .with_columns(
            pl.when(pl.col(bucket_col) == bucket_config.low_volatility_bucket)
            .then(pl.lit("long"))
            .otherwise(pl.lit("short"))
            .alias("leg")
        )
        .rename({date_col: "signal_date"})
    )

    low_long = _equal_weight_scenario(
        candidates.filter(pl.col("leg") == "long"),
        scenario=scenario_config.low_volatility_long,
        signed=False,
    )
    high_long = _equal_weight_scenario(
        candidates.filter(pl.col("leg") == "short"),
        scenario=scenario_config.high_volatility_long,
        signed=False,
    )
    naive_ls = _equal_weight_scenario(
        candidates,
        scenario=scenario_config.naive_equal_weight_long_short,
        signed=True,
    )

    group_columns = ["signal_date", "leg"]
    vol_scaled = (
        candidates.with_columns(pl.len().over(group_columns).alias("leg_size"))
        .with_columns(
            (
                (1.0 / pl.col("leg_size"))
                * sizing_config.annualized_stock_volatility_target
                / pl.col(sizing_col)
            )
            .clip(upper_bound=sizing_config.maximum_absolute_stock_weight)
            .alias("raw_absolute_weight")
        )
        .with_columns(
            pl.col("raw_absolute_weight")
            .sum()
            .over(group_columns)
            .alias("raw_leg_gross")
        )
        .with_columns(
            (
                pl.col("raw_absolute_weight")
                * pl.when(
                    pl.col("raw_leg_gross") > sizing_config.maximum_leg_gross_exposure
                )
                .then(
                    sizing_config.maximum_leg_gross_exposure / pl.col("raw_leg_gross")
                )
                .otherwise(1.0)
                * pl.when(pl.col("leg") == "long").then(1.0).otherwise(-1.0)
            ).alias("weight")
        )
        .with_columns(
            pl.lit(scenario_config.volatility_scaled_long_short).alias("scenario")
        )
    )

    columns = [
        "signal_date",
        asset_col,
        "scenario",
        "leg",
        "weight",
        "stock_beta",
        sizing_col,
        bucket_col,
    ]
    return pl.concat(
        [
            low_long.select(columns),
            high_long.select(columns),
            naive_ls.select(columns),
            vol_scaled.select(columns),
        ],
        how="vertical",
    ).collect()


def build_decile_targets(
    signal_snapshots: Frame,
    data_config: DataConfig,
    bucket_config: BucketConfig,
) -> pl.DataFrame:
    """Build fully invested positive-weight portfolios for every volatility decile."""

    frame = as_lazy(signal_snapshots)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    bucket_col = bucket_config.bucket_column
    require_columns(frame, [date_col, asset_col, bucket_col], "signal_snapshots")
    return (
        frame.drop_nulls(bucket_col)
        .rename({date_col: "signal_date"})
        .with_columns(
            pl.format("decile_{}", pl.col(bucket_col)).alias("scenario"),
            pl.len().over(["signal_date", bucket_col]).alias("bucket_size"),
        )
        .with_columns((1.0 / pl.col("bucket_size")).alias("weight"))
        .select("signal_date", asset_col, "scenario", "weight")
        .collect()
    )


def summarize_target_exposures(targets: pl.DataFrame) -> pl.DataFrame:
    return (
        targets.group_by("signal_date", "scenario")
        .agg(
            pl.col("weight").abs().sum().alias("gross_exposure"),
            pl.col("weight").sum().alias("net_exposure"),
            pl.col("weight").filter(pl.col("weight") > 0).sum().alias("long_exposure"),
            (-pl.col("weight").filter(pl.col("weight") < 0).sum()).alias(
                "short_exposure"
            ),
            (pl.col("weight") * pl.col("stock_beta")).sum().alias("stock_beta"),
            pl.len().alias("positions"),
        )
        .sort("signal_date", "scenario")
    )
