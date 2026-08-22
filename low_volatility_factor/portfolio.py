"""Portfolio selection and target-weight construction."""

from __future__ import annotations

from collections.abc import Sequence

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


def _capped_proportional_weights(
    market_caps: list[float],
    maximum_weight: float,
) -> list[float]:
    """Return proportional weights with a cap and exact unit gross exposure."""

    if not market_caps or any(value <= 0 for value in market_caps):
        raise ValueError("market caps must be positive")
    if not 0 < maximum_weight <= 1:
        raise ValueError("maximum_weight must be in (0, 1]")
    if len(market_caps) * maximum_weight < 1:
        raise ValueError("maximum_weight is too small for the portfolio size")

    total_market_cap = sum(market_caps)
    proportions = [value / total_market_cap for value in market_caps]
    weights = [0.0] * len(proportions)
    remaining = set(range(len(proportions)))
    remaining_gross = 1.0

    while remaining:
        remaining_market_cap = sum(proportions[index] for index in remaining)
        scale = remaining_gross / remaining_market_cap
        capped = [
            index
            for index in remaining
            if proportions[index] * scale > maximum_weight
        ]
        if not capped:
            for index in remaining:
                weights[index] = proportions[index] * scale
            break
        for index in capped:
            weights[index] = maximum_weight
            remaining.remove(index)
            remaining_gross -= maximum_weight

    return weights


def _market_cap_weight_scenario(
    candidates: pl.DataFrame,
    *,
    market_cap_column: str,
    maximum_weight: float,
    scenario: str,
    signed: bool,
    group_columns: Sequence[str] = ("signal_date", "leg"),
) -> pl.DataFrame:
    """Build capped market-cap weights, optionally signed by long/short leg."""

    pieces: list[pl.DataFrame] = []
    for group in candidates.partition_by(list(group_columns), maintain_order=True):
        weights = _capped_proportional_weights(
            group.get_column(market_cap_column).cast(pl.Float64).to_list(),
            maximum_weight,
        )
        weighted = group.with_columns(
            pl.Series("weight", weights, dtype=pl.Float64)
        )
        if signed:
            weighted = weighted.with_columns(
                (
                    pl.when(pl.col("leg") == "long")
                    .then(pl.col("weight"))
                    .otherwise(-pl.col("weight"))
                ).alias("weight")
            )
        pieces.append(weighted.with_columns(pl.lit(scenario).alias("scenario")))
    return pl.concat(pieces, how="vertical")


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


def build_stage_targets(
    signal_snapshots: Frame,
    data_config: DataConfig,
    signal_config: SignalConfig,
    bucket_config: BucketConfig,
    sizing_config: SizingConfig,
    scenario_config: ScenarioConfig,
) -> pl.DataFrame:
    """Build market-cap reference portfolios and a market-cap inverse-vol portfolio."""

    frame = as_lazy(signal_snapshots)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    bucket_col = bucket_config.bucket_column
    sizing_col = signal_config.sizing_volatility_column
    required = [
        date_col,
        asset_col,
        bucket_col,
        sizing_col,
        data_config.market_cap_column,
        "stock_beta",
    ]
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
        .drop_nulls([sizing_col, data_config.market_cap_column, "stock_beta"])
        .with_columns(
            pl.when(pl.col(bucket_col) == bucket_config.low_volatility_bucket)
            .then(pl.lit("long"))
            .otherwise(pl.lit("short"))
            .alias("leg")
        )
        .rename({date_col: "signal_date"})
        .collect()
    )

    low_long = _market_cap_weight_scenario(
        candidates.filter(pl.col("leg") == "long"),
        market_cap_column=data_config.market_cap_column,
        maximum_weight=sizing_config.maximum_absolute_stock_weight,
        scenario=scenario_config.low_volatility_long,
        signed=False,
    )
    high_long = _market_cap_weight_scenario(
        candidates.filter(pl.col("leg") == "short"),
        market_cap_column=data_config.market_cap_column,
        maximum_weight=sizing_config.maximum_absolute_stock_weight,
        scenario=scenario_config.high_volatility_long,
        signed=False,
    )
    naive_ls = _market_cap_weight_scenario(
        candidates,
        market_cap_column=data_config.market_cap_column,
        maximum_weight=sizing_config.maximum_absolute_stock_weight,
        scenario=scenario_config.market_cap_long_short,
        signed=True,
    )

    group_columns = ["signal_date", "leg"]
    vol_scaled = (
        candidates.lazy().with_columns(
            pl.col(data_config.market_cap_column)
            .sum()
            .over(group_columns)
            .alias("leg_market_cap")
        )
        .with_columns(
            (
                (pl.col(data_config.market_cap_column) / pl.col("leg_market_cap"))
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
        .collect()
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
    )


def build_decile_targets(
    signal_snapshots: Frame,
    data_config: DataConfig,
    bucket_config: BucketConfig,
    maximum_weight: float,
) -> pl.DataFrame:
    """Build fully invested positive-weight portfolios for every volatility decile."""

    frame = as_lazy(signal_snapshots)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    bucket_col = bucket_config.bucket_column
    require_columns(
        frame,
        [date_col, asset_col, bucket_col, data_config.market_cap_column],
        "signal_snapshots",
    )
    candidates = (
        frame.drop_nulls(bucket_col)
        .rename({date_col: "signal_date"})
        .collect()
    )
    pieces: list[pl.DataFrame] = []
    for bucket in candidates.get_column(bucket_col).unique().sort().to_list():
        bucket_candidates = candidates.filter(pl.col(bucket_col) == bucket)
        weighted = _market_cap_weight_scenario(
            bucket_candidates.with_columns(pl.lit("long").alias("leg")),
            market_cap_column=data_config.market_cap_column,
            maximum_weight=maximum_weight,
            scenario=f"decile_{bucket}",
            signed=False,
            group_columns=("signal_date", "leg"),
        )
        pieces.append(weighted.select("signal_date", asset_col, "scenario", "weight"))
    return pl.concat(pieces, how="vertical")


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
