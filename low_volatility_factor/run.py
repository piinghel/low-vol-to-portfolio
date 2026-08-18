"""One-command runner for the low-volatility factor research."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

from .backtest import (
    build_execution_schedule,
    map_dates_to_signal_periods,
    prepare_held_returns,
    simulate_stock_targets,
)
from .config import DataConfig, ResearchConfig
from .data import (
    align_prices_to_market_calendar,
    build_investable_universe,
    load_constituent_data,
    load_price_data,
    prepare_index_returns,
    resolve_input_files,
)
from .frames import require_finite_float
from .metrics import compute_simple_metrics, compute_stage_metrics
from .plots import render_article_figures
from .portfolio import (
    build_decile_targets,
    build_stage_targets,
    select_rebalance_signal_dates,
    summarize_target_exposures,
)
from .risk import compute_trailing_market_beta
from .signals import (
    assign_volatility_buckets,
    compute_selection_volatility,
    compute_sizing_volatility,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_manifest(config: ResearchConfig) -> dict[str, object]:
    patterns = {
        "prices": config.data.price_glob,
        "constituents": config.data.constituents_glob,
        "index": config.data.index_glob,
    }
    files: dict[str, list[dict[str, object]]] = {}
    for label, pattern in patterns.items():
        files[label] = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=UTC
                ).isoformat(),
            }
            for path in resolve_input_files(config.data.data_root, pattern)
        ]
    return {
        "generated_utc": datetime.now(tz=UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "polars": pl.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "files": files,
    }


def _collect_data_quality(
    raw_prices: pl.LazyFrame,
    aligned_prices: pl.LazyFrame,
    config: ResearchConfig,
) -> pl.DataFrame:
    """Summarize raw inputs and the market-calendar normalization."""

    return_column = config.data.total_return_column
    raw_quality = (
        raw_prices.select(
            pl.len().alias("raw_price_rows"),
            pl.col(config.data.date_column).n_unique().alias("raw_distinct_dates"),
            pl.col(return_column).null_count().alias("raw_missing_total_returns"),
        )
        .collect()
        .row(0, named=True)
    )
    aligned_quality = (
        aligned_prices.select(
            pl.len().alias("market_aligned_price_rows"),
            pl.col(config.data.date_column).n_unique().alias("market_dates"),
            ((pl.col(return_column) - pl.col("source_total_return")).abs() > 1e-12)
            .sum()
            .alias("returns_changed_by_calendar_alignment"),
            (pl.col(return_column).abs() > 1.0)
            .sum()
            .alias("aligned_returns_over_100_percent"),
            pl.col(return_column).abs().max().alias("maximum_absolute_aligned_return"),
        )
        .collect()
        .row(0, named=True)
    )
    return pl.DataFrame(
        [
            {"metric": key, "value": str(value)}
            for key, value in {**raw_quality, **aligned_quality}.items()
        ]
    )


def _build_signal_snapshots(
    prices: pl.LazyFrame,
    constituents: pl.LazyFrame,
    market: pl.DataFrame,
    config: ResearchConfig,
) -> pl.DataFrame:
    """Build complete point-in-time signal cross-sections at each rebalance."""

    features = compute_selection_volatility(prices, config.data, config.signal)
    features = compute_sizing_volatility(
        features, config.data, config.signal, config.sizing
    )
    betas = compute_trailing_market_beta(
        prices,
        market,
        config.data,
        config.beta,
    )
    universe = build_investable_universe(
        features.join(
            betas,
            on=[config.data.date_column, config.data.asset_column],
            how="left",
        ),
        constituents,
        config.data,
    )
    signal_dates = select_rebalance_signal_dates(market, config.data, config.backtest)
    snapshots = (
        universe.join(signal_dates.lazy(), on=config.data.date_column, how="inner")
        .select(
            config.data.date_column,
            config.data.asset_column,
            config.signal.signal_column,
            config.signal.sizing_volatility_column,
            "stock_beta",
        )
        .drop_nulls(
            [
                config.signal.signal_column,
                config.signal.sizing_volatility_column,
                "stock_beta",
            ]
        )
        .pipe(
            assign_volatility_buckets,
            config.data,
            config.signal,
            config.buckets,
        )
        .collect(engine="streaming")
    )
    if snapshots.is_empty():
        raise RuntimeError("No valid signal snapshots were produced")
    return snapshots


def _summarize_sample(
    signal_snapshots: pl.DataFrame,
    config: ResearchConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return rebalance universe sizes and their persisted summary table."""

    date_column = config.data.date_column
    asset_column = config.data.asset_column
    snapshot_sizes = (
        signal_snapshots.group_by(date_column)
        .len()
        .rename({"len": "eligible_stocks"})
        .sort(date_column)
    )
    counts = snapshot_sizes.get_column("eligible_stocks")
    summary = {
        "first_signal_date": signal_snapshots.get_column(date_column).min(),
        "last_signal_date": signal_snapshots.get_column(date_column).max(),
        "rebalance_dates": signal_snapshots.get_column(date_column).n_unique(),
        "unique_eligible_stocks": signal_snapshots.get_column(asset_column).n_unique(),
        "minimum_rebalance_cross_section": counts.min(),
        "median_rebalance_cross_section": counts.median(),
        "maximum_rebalance_cross_section": counts.max(),
    }
    summary_table = pl.DataFrame(
        [{"metric": key, "value": str(value)} for key, value in summary.items()]
    )
    return snapshot_sizes, summary_table


def _select_asset_prices(
    prices: pl.LazyFrame,
    stage_targets: pl.DataFrame,
    decile_targets: pl.DataFrame,
    config: ResearchConfig,
) -> pl.DataFrame:
    """Collect adjusted prices and vendor returns for targeted assets."""

    asset_column = config.data.asset_column
    target_assets = (
        pl.concat(
            [
                stage_targets.select(asset_column),
                decile_targets.select(asset_column),
            ]
        )
        .unique()
        .get_column(asset_column)
        .to_list()
    )
    return (
        prices.filter(pl.col(asset_column).is_in(target_assets))
        .select(
            config.data.date_column,
            asset_column,
            config.data.adjusted_price_column,
            config.data.total_return_column,
        )
        .collect(engine="streaming")
    )


def _collect_held_return_quality(
    date_to_signal: pl.DataFrame,
    stage_targets: pl.DataFrame,
    asset_returns: pl.DataFrame,
    config: ResearchConfig,
) -> pl.DataFrame:
    """Summarize and enforce return quality for assets actually held."""

    return_column = config.data.total_return_column
    quality = (
        prepare_held_returns(
            stage_targets,
            asset_returns,
            date_to_signal,
            config.data,
        )
        .group_by("scenario")
        .agg(
            pl.len().alias("position_return_observations"),
            pl.col(return_column).null_count().alias("missing_returns"),
            (pl.col(return_column).abs() > 1.0).sum().alias("returns_over_100_percent"),
            pl.col(return_column).abs().max().alias("maximum_absolute_return"),
        )
        .sort("scenario")
    )
    maximum_value = quality.get_column("maximum_absolute_return").max()
    if maximum_value is None:
        raise RuntimeError("Held-return quality gate failed: no finite returns found")
    maximum_held_return = require_finite_float(
        maximum_value, "maximum absolute held return"
    )
    threshold = config.data.maximum_held_absolute_daily_return
    if threshold is not None and maximum_held_return > threshold:
        raise RuntimeError(
            "Held-return quality gate failed: maximum absolute return "
            f"{maximum_held_return:.2%} exceeds "
            f"{config.data.maximum_held_absolute_daily_return:.2%}"
        )
    return quality


def run(
    config: ResearchConfig,
    output_directory: Path,
) -> None:
    output = output_directory.expanduser().resolve()
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    obsolete_paths = [
        output / "hedge_targets.csv",
        output / "missing_returns.csv",
        figures / "beta_hedge.png",
    ]
    for stem in ("cumulative_performance", "drawdowns", "dotcom_comparison"):
        obsolete_paths.extend(
            figures / f"{stem}{suffix}"
            for suffix in (".png", "_mobile.png", ".svg", "_mobile.svg")
        )
    for obsolete_path in obsolete_paths:
        obsolete_path.unlink(missing_ok=True)

    _write_json(output / "config.json", asdict(config))
    _write_json(output / "data_manifest.json", _input_manifest(config))

    raw_prices = load_price_data(config.data).select(
        config.data.date_column,
        config.data.asset_column,
        config.data.adjusted_price_column,
        config.data.unadjusted_price_column,
        config.data.total_return_column,
    )
    price_bounds = (
        raw_prices.select(
            pl.col(config.data.date_column).min().alias("first_price_date"),
            pl.col(config.data.date_column).max().alias("last_price_date"),
        )
        .collect()
        .row(0, named=True)
    )
    if any(value is None for value in price_bounds.values()):
        raise ValueError("Price data contains no usable dates")
    market = (
        prepare_index_returns(config.data)
        .filter(
            pl.col(config.data.date_column).is_between(
                price_bounds["first_price_date"],
                price_bounds["last_price_date"],
            )
        )
        .collect()
    )
    prices = align_prices_to_market_calendar(raw_prices, market, config.data)
    constituents = load_constituent_data(config.data)

    _collect_data_quality(raw_prices, prices, config).write_csv(
        output / "data_quality.csv"
    )
    signal_snapshots = _build_signal_snapshots(
        prices,
        constituents,
        market,
        config,
    )
    snapshot_sizes, sample_summary = _summarize_sample(signal_snapshots, config)
    sample_summary.write_csv(output / "sample_summary.csv")
    snapshot_sizes.write_csv(output / "eligible_universe.csv")

    stage_targets = build_stage_targets(
        signal_snapshots,
        config.data,
        config.signal,
        config.buckets,
        config.sizing,
        config.scenarios,
    )
    decile_targets = build_decile_targets(signal_snapshots, config.data, config.buckets)
    target_exposures = summarize_target_exposures(stage_targets)

    valid_signal_dates = (
        stage_targets.select("signal_date").unique().sort("signal_date")
    )
    schedule = build_execution_schedule(
        valid_signal_dates.rename({"signal_date": config.data.date_column}),
        market,
        config.data,
        config.backtest,
    )
    date_to_signal = map_dates_to_signal_periods(schedule, market, config.data)

    asset_prices = _select_asset_prices(
        prices,
        stage_targets,
        decile_targets,
        config,
    )
    held_return_quality = _collect_held_return_quality(
        date_to_signal,
        stage_targets,
        asset_prices,
        config,
    )
    held_return_quality.write_csv(output / "held_return_quality.csv")

    stage_daily = simulate_stock_targets(
        stage_targets,
        asset_prices,
        date_to_signal,
        schedule,
        config.data,
        config.costs,
    )
    scaled_leg_targets = stage_targets.filter(
        pl.col("scenario") == config.scenarios.volatility_scaled_long_short
    ).with_columns(
        pl.when(pl.col("leg") == "long")
        .then(pl.lit("scaled_long_leg"))
        .otherwise(pl.lit("scaled_short_leg"))
        .alias("scenario")
    )
    scaled_leg_daily = simulate_stock_targets(
        scaled_leg_targets,
        asset_prices,
        date_to_signal,
        schedule,
        config.data,
        config.costs,
    )

    decile_targets = decile_targets.with_columns(pl.lit(0.0).alias("stock_beta"))
    decile_daily = simulate_stock_targets(
        decile_targets,
        asset_prices,
        date_to_signal,
        schedule,
        config.data,
        config.costs,
    )
    decile_metrics = compute_simple_metrics(
        decile_daily,
        annualization=config.backtest.annualization_factor,
    )
    stage_metrics = compute_stage_metrics(
        stage_daily, annualization=config.backtest.annualization_factor
    )

    stage_targets.write_parquet(output / "stage_targets.parquet")
    target_exposures.write_parquet(output / "target_exposures.parquet")
    target_exposures.write_csv(output / "target_exposures.csv")
    schedule.write_csv(output / "execution_schedule.csv")
    stage_daily.write_parquet(output / "daily_stage_results.parquet")
    scaled_leg_daily.write_parquet(output / "daily_scaled_leg_results.parquet")
    stage_metrics.write_csv(output / "stage_metrics.csv")
    stage_metrics.write_parquet(output / "stage_metrics.parquet")
    decile_metrics.write_csv(output / "decile_metrics.csv")

    render_article_figures(
        figures,
        snapshot_sizes=snapshot_sizes,
        decile_metrics=decile_metrics,
        stage_metrics=stage_metrics,
        stage_daily=stage_daily,
        scaled_leg_daily=scaled_leg_daily,
        config=config,
    )

    print(f"Wrote low-volatility research artifacts to {output}")
    print(stage_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Input export directory (default: "
            "<Documents>/research_data/riy_backtest_data_20260818)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output directory (default: <project-root>/output/latest)"),
    )
    args = parser.parse_args()
    article_project = Path(__file__).resolve().parents[1]
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else Path(__file__).resolve().parents[3]
        / "research_data"
        / "riy_backtest_data_20260818"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else article_project / "output" / "latest"
    )
    config = ResearchConfig(data=DataConfig(data_root=data_root))
    run(config, output)


if __name__ == "__main__":
    main()
