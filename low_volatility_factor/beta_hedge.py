"""Replay causal index hedges from saved signal-date portfolio betas."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from . import backtest, beta_comparison, config, metrics


def hedge_daily(
    daily: pl.DataFrame, targets: pl.DataFrame, schedule: pl.DataFrame
) -> pl.DataFrame:
    """Use signal-close estimates, next-close trades and fixed index quantities."""
    calendar = (
        daily.select("date", "signal_date", "market_return").unique().sort("date")
    )
    if calendar.get_column("date").n_unique() != calendar.height:
        raise ValueError("Rules must share one market calendar and signal schedule")
    if schedule.filter(
        (pl.col("signal_date") >= pl.col("execution_date"))
        | (pl.col("execution_date") >= pl.col("effective_return_date"))
    ).height:
        raise ValueError("Signal must precede execution and first earned return")
    expected = calendar.join(schedule, on="signal_date", validate="m:1")
    if (
        expected.height != calendar.height
        or expected.filter(pl.col("date") < pl.col("effective_return_date")).height
    ):
        raise ValueError("Daily signal mapping violates execution timing")
    initial_date = schedule.get_column("execution_date").min()
    prices = pl.concat(
        [
            pl.DataFrame({"date": [initial_date], "px_last": [1.0]}),
            calendar.select(
                "date",
                (1 + pl.col("market_return").cast(pl.Float64))
                .cum_prod()
                .alias("px_last"),
            ),
        ]
    ).with_columns(pl.lit("MARKET_INDEX").alias("asset_id_bb_global"))
    hedge_targets = targets.filter(
        pl.col("scenario").is_in(beta_comparison.RULES)
    ).select(
        "signal_date",
        "scenario",
        (-pl.col("stock_beta")).alias("weight"),
        pl.lit(1.0).alias("stock_beta"),
        pl.lit("MARKET_INDEX").alias("asset_id_bb_global"),
    )
    if (
        hedge_targets.select(pl.struct("signal_date", "scenario").n_unique()).item()
        != hedge_targets.height
    ):
        raise ValueError("Duplicate hedge targets")
    overlay = backtest.simulate_stock_targets(
        hedge_targets,
        prices,
        calendar,
        schedule,
        config.DataConfig(data_root=Path(".")),
        config.CostConfig(equity_cost_bps=5),
    ).select(
        "date",
        "scenario",
        pl.col("gross_return").alias("hedge_gross_return"),
        pl.col("equity_cost").alias("hedge_cost"),
        pl.col("equity_turnover").alias("hedge_turnover"),
        (pl.col("net_exposure") / (1 + pl.col("market_return"))).alias(
            "hedge_start_exposure"
        ),
    )
    days = prices.select(
        "date", pl.col("date").diff().dt.total_days().alias("calendar_days")
    )
    result = daily.join(overlay, on=["date", "scenario"], validate="1:1").join(
        days, on="date"
    )
    if result.height != daily.height or result.null_count().sum_horizontal().sum():
        raise ValueError("Incomplete hedge replay")
    np.testing.assert_allclose(
        result["hedge_cost"], result["hedge_turnover"] * 0.0005, atol=1e-12
    )
    np.testing.assert_allclose(
        result["hedge_gross_return"],
        result["hedge_start_exposure"] * result["market_return"],
        atol=1e-10,
    )
    return result.sort("scenario", "date")


def evaluate(daily: pl.DataFrame) -> pl.DataFrame:
    """Report unchanged originals and prespecified financing sensitivities."""
    rows = []
    for rule in beta_comparison.RULES:
        part = daily.filter(pl.col("scenario") == rule).sort("date")
        for funding in (None, 0.0, 0.03, 0.05):
            charge = (
                np.zeros(part.height)
                if funding is None
                else (
                    part["hedge_start_exposure"] * part["calendar_days"] * funding / 365
                ).to_numpy()
            )
            returns = part["net_return"].to_numpy().copy()
            if funding is not None:
                returns += (
                    part["hedge_gross_return"].to_numpy()
                    - part["hedge_cost"].to_numpy()
                    - charge
                )
            rows.append(
                {
                    "scenario": rule,
                    "treatment": "original" if funding is None else "hedged",
                    "funding_rate": funding,
                    **metrics._metric_row(returns, annualization=252),
                    "realized_net_beta": beta_comparison.realized_beta(
                        returns, part["market_return"].to_numpy()
                    ),
                    "annual_stock_turnover": float(
                        part["equity_turnover"].to_numpy().mean() * 252
                    ),
                    "annual_hedge_turnover": 0.0
                    if funding is None
                    else float(part["hedge_turnover"].to_numpy().mean() * 252),
                    "annual_funding_cost": float(charge.mean() * 252),
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    paths = {
        name: args.input / name
        for name in (
            "stage_daily.parquet",
            "stage_metrics.csv",
            "target_exposures.csv",
            "execution_schedule.csv",
        )
    }
    original = pl.read_parquet(paths["stage_daily.parquet"]).filter(
        pl.col("scenario").is_in(beta_comparison.RULES)
    )
    # Reuse the historical reconciliation; its fitted slopes never enter hedge sizing.
    beta_comparison.compare(original, pl.read_csv(paths["stage_metrics.csv"]))
    daily = hedge_daily(
        original,
        pl.read_csv(paths["target_exposures.csv"], try_parse_dates=True),
        pl.read_csv(paths["execution_schedule.csv"], try_parse_dates=True),
    )
    results = evaluate(daily)
    args.output.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(args.output / "daily.parquet")
    results.write_csv(args.output / "metrics.csv")
    summary = {
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths.values()
        },
        "versions": {"polars": pl.__version__, "numpy": np.__version__},
        "method": "Signal-date target beta, next-close execution, fixed quantities; 5bp hedge trades; 3% annual signed prior-close notional financing ACT/365 with daily PnL swept, 0%/5% sensitivities. Original equity financing remains unchanged. Synthetic saved index proxy, no historical futures basis/roll or funding vintage claims.",
        "checks": "Original metrics reconcile; strict signal/execution/return order; complete joins; package hedge PnL and 5bp turnover-cost identities.",
        "metrics": results.to_dicts(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results)


if __name__ == "__main__":
    main()
