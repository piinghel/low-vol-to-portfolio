"""Strictly trailing volatility signals and deterministic cross-sectional buckets."""

from __future__ import annotations

import math

import polars as pl

from .config import BucketConfig, DataConfig, SignalConfig, SizingConfig
from .frames import Frame, as_lazy, require_columns


def compute_selection_volatility(
    universe: Frame,
    data_config: DataConfig,
    signal_config: SignalConfig,
) -> pl.LazyFrame:
    """Compute the mean annualized trailing volatility across configured windows."""

    frame = as_lazy(universe)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    price_col = data_config.adjusted_price_column
    return_col = signal_config.return_column

    require_columns(frame, [date_col, asset_col, price_col], "universe")
    frame = frame.sort([asset_col, date_col]).with_columns(
        pl.col(price_col)
        .pct_change()
        .over(asset_col, order_by=date_col)
        .alias(return_col)
    )

    annualization = math.sqrt(signal_config.annualization_factor)
    volatility_columns: list[str] = []
    expressions: list[pl.Expr] = []
    for window in signal_config.windows:
        column = f"volatility_{window}d"
        volatility_columns.append(column)
        expressions.append(
            (
                pl.col(return_col)
                .rolling_std(window_size=window, min_samples=window)
                .over(asset_col, order_by=date_col)
                * annualization
            ).alias(column)
        )

    return (
        frame.with_columns(expressions)
        .with_columns(
            pl.mean_horizontal(volatility_columns).alias("selection_volatility_raw")
        )
        .with_columns(
            pl.col("selection_volatility_raw")
            .clip(
                signal_config.minimum_annualized_volatility,
                signal_config.maximum_annualized_volatility,
            )
            .alias(signal_config.signal_column)
        )
    )


def compute_sizing_volatility(
    signals: Frame,
    data_config: DataConfig,
    signal_config: SignalConfig,
    sizing_config: SizingConfig,
) -> pl.LazyFrame:
    """Add the separate trailing volatility estimate used only for position sizing."""

    frame = as_lazy(signals)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    return_col = signal_config.return_column
    require_columns(frame, [date_col, asset_col, return_col], "signals")

    return frame.with_columns(
        (
            pl.col(return_col)
            .rolling_std(
                window_size=sizing_config.volatility_window,
                min_samples=sizing_config.volatility_window,
            )
            .over(asset_col, order_by=date_col)
            * math.sqrt(signal_config.annualization_factor)
        )
        .clip(lower_bound=signal_config.minimum_annualized_volatility)
        .alias(signal_config.sizing_volatility_column)
    )


def assign_volatility_buckets(
    signals: Frame,
    data_config: DataConfig,
    signal_config: SignalConfig,
    bucket_config: BucketConfig,
) -> pl.LazyFrame:
    """Assign approximately equal-sized deciles, with low volatility in bucket 1."""

    frame = as_lazy(signals)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    signal_col = signal_config.signal_column
    bucket_col = bucket_config.bucket_column
    number_of_buckets = bucket_config.number_of_buckets

    require_columns(frame, [date_col, asset_col, signal_col], "signals")
    ranked = (
        frame.drop_nulls(signal_col)
        .sort([date_col, signal_col, asset_col])
        .with_columns(
            pl.col(signal_col)
            .rank(method="ordinal")
            .over(date_col)
            .alias("cross_section_rank"),
            pl.len().over(date_col).alias("cross_section_size"),
        )
    )

    bucket = (
        (pl.col("cross_section_rank") - 1)
        * number_of_buckets
        / pl.col("cross_section_size")
    ).floor().cast(pl.Int8) + 1
    return ranked.with_columns(bucket.clip(1, number_of_buckets).alias(bucket_col))
