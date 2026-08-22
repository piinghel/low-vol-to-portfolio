"""Data loading and causal point-in-time universe construction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .config import DataConfig
from .frames import Frame, as_lazy, require_columns


def resolve_input_files(data_root: Path, pattern: str) -> list[Path]:
    paths = sorted(data_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {data_root / pattern}")
    return paths


def scan_parquet_partitions(paths: Sequence[Path]) -> pl.LazyFrame:
    if not paths:
        raise ValueError("At least one parquet path is required")
    return pl.concat(
        [pl.scan_parquet(path) for path in paths],
        how="diagonal_relaxed",
    )


def load_price_data(config: DataConfig) -> pl.LazyFrame:
    return scan_parquet_partitions(
        resolve_input_files(config.data_root, config.price_glob)
    )


def load_constituent_data(config: DataConfig) -> pl.LazyFrame:
    return scan_parquet_partitions(
        resolve_input_files(config.data_root, config.constituents_glob)
    )


def load_index_data(config: DataConfig) -> pl.LazyFrame:
    return scan_parquet_partitions(
        resolve_input_files(config.data_root, config.index_glob)
    )


def prepare_index_returns(config: DataConfig) -> pl.LazyFrame:
    """Return the reference-index calendar and close-to-close return series."""

    return (
        load_index_data(config)
        .select(config.date_column, config.index_price_column)
        .unique(config.date_column)
        .sort(config.date_column)
        .with_columns(
            pl.col(config.index_price_column).pct_change().alias("market_return")
        )
    )


def align_prices_to_market_calendar(
    prices: Frame,
    market_calendar: Frame,
    config: DataConfig,
) -> pl.LazyFrame:
    """Compound source total returns across non-market rows, then sample market dates.

    A few source partitions contain weekend or holiday stock observations that are not
    present in the reference-index calendar. Using the vendor's row-level return on the
    next market date can then retain only one side of a corporate-action round trip.
    Building a cumulative total-return index before calendar alignment preserves the
    compounded return across those intervening rows.
    """

    frame = as_lazy(prices)
    calendar = as_lazy(market_calendar)
    date_col = config.date_column
    asset_col = config.asset_column
    return_col = config.total_return_column
    require_columns(
        frame,
        [
            date_col,
            asset_col,
            config.adjusted_price_column,
            config.unadjusted_price_column,
            return_col,
        ],
        "prices",
    )
    require_columns(calendar, [date_col], "market_calendar")

    return (
        frame.sort([asset_col, date_col])
        .with_columns(
            pl.col(return_col).alias("source_total_return"),
            pl.col(return_col).is_null().alias("source_return_missing"),
            (1.0 + pl.col(return_col).fill_null(0.0))
            .cum_prod()
            .over(asset_col, order_by=date_col)
            .alias("_total_return_index"),
        )
        .join(calendar.select(date_col).unique(), on=date_col, how="inner")
        .sort([asset_col, date_col])
        .with_columns(
            pl.col("_total_return_index")
            .pct_change()
            .over(asset_col, order_by=date_col)
            .alias(return_col)
        )
        .drop("_total_return_index")
    )


def build_investable_universe(
    prices: Frame,
    constituents: Frame,
    config: DataConfig,
) -> pl.LazyFrame:
    """Match every price date to the latest constituent snapshot available by then.

    Constituent files contain complete cross-sectional snapshots. Mapping price dates to
    snapshots before joining assets correctly represents removals as well as additions.
    The price threshold is applied to unadjusted price on the observation date.
    """

    price_lf = as_lazy(prices)
    constituent_lf = as_lazy(constituents)
    date_col = config.date_column
    asset_col = config.asset_column
    snapshot_col = "constituent_snapshot_date"

    require_columns(
        price_lf,
        [
            date_col,
            asset_col,
            config.adjusted_price_column,
            config.unadjusted_price_column,
        ],
        "prices",
    )
    require_columns(constituent_lf, [date_col, asset_col], "constituents")

    snapshot_members = (
        constituent_lf.select(
            pl.col(date_col).alias(snapshot_col),
            pl.col(asset_col),
        )
        .drop_nulls()
        .unique([snapshot_col, asset_col])
    )
    snapshot_calendar = (
        snapshot_members.select(snapshot_col).unique().sort(snapshot_col)
    )
    price_calendar = price_lf.select(date_col).unique().sort(date_col)
    price_to_snapshot = price_calendar.join_asof(
        snapshot_calendar,
        left_on=date_col,
        right_on=snapshot_col,
        strategy="backward",
    ).drop_nulls(snapshot_col)

    return (
        price_lf.join(price_to_snapshot, on=date_col, how="inner")
        .join(snapshot_members, on=[snapshot_col, asset_col], how="inner")
        .filter(
            pl.col(config.unadjusted_price_column) >= config.minimum_unadjusted_price
        )
        .sort([asset_col, date_col])
    )
