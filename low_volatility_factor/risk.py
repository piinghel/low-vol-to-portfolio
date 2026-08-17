"""Trailing market-beta estimation for portfolio diagnostics."""

from __future__ import annotations

import polars as pl

from .config import BetaConfig, DataConfig
from .frames import Frame, as_lazy, require_columns


def compute_trailing_market_beta(
    asset_returns: Frame,
    market_returns: Frame,
    data_config: DataConfig,
    beta_config: BetaConfig,
) -> pl.LazyFrame:
    """Estimate trailing covariance beta relative to the market."""

    assets = as_lazy(asset_returns)
    market = as_lazy(market_returns)
    date_col = data_config.date_column
    asset_col = data_config.asset_column
    return_col = data_config.total_return_column

    require_columns(assets, [date_col, asset_col, return_col], "asset_returns")
    require_columns(market, [date_col, "market_return"], "market_returns")

    market_with_variance = (
        market.select(date_col, "market_return")
        .sort(date_col)
        .with_columns(
            pl.col("market_return")
            .rolling_var(
                window_size=beta_config.lookback,
                min_samples=beta_config.minimum_observations,
            )
            .alias("market_variance")
        )
    )

    return (
        assets.select(date_col, asset_col, return_col)
        .sort([asset_col, date_col])
        .join(market_with_variance, on=date_col, how="left")
        .with_columns(
            pl.rolling_cov(
                pl.col(return_col),
                pl.col("market_return"),
                window_size=beta_config.lookback,
                min_samples=beta_config.minimum_observations,
            )
            .over(asset_col, order_by=date_col)
            .alias("market_covariance")
        )
        .with_columns(
            pl.when(
                pl.col("market_covariance").is_finite()
                & pl.col("market_variance").is_finite()
                & (pl.col("market_variance") > 0)
            )
            .then(
                (pl.col("market_covariance") / pl.col("market_variance")).clip(
                    beta_config.beta_clip[0], beta_config.beta_clip[1]
                )
            )
            .otherwise(None)
            .alias("stock_beta")
        )
        .select(date_col, asset_col, "stock_beta")
    )
