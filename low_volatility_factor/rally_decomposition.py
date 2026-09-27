"""Split each scaled book's rally P&L into its market-beta part and a residual.

In a rally the short book's gross contribution is always negative, and the
inverse-volatility sizing gives it more market beta per dollar by design, so
gross contributions alone overstate its role. This removes each book's own
market exposure, measured two ways: the book's ex-ante stock beta on each day,
and its in-window OLS beta. Contributions add daily before-cost P&L per unit of
strategy notional, in percentage points.

    uv run python -m low_volatility_factor.rally_decomposition \\
        --input output/turnover-review-2026-09-05 --output output/rally-decomposition-2026-09-27
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .beta_comparison import realized_beta

# Article windows: each starts at the portfolio's high before the rally.
EPISODES = {
    "dot-com": (date(1998, 10, 8), date(2000, 3, 9)),
    "2025-26": (date(2025, 4, 3), date(2026, 5, 27)),
}
BOOKS = {"long": "scaled_long_leg", "short": "scaled_short_leg"}


def decompose(legs: pl.DataFrame) -> pl.DataFrame:
    """One row per episode and book: total, beta parts and residuals, in points."""
    rows = []
    for episode, (start, end) in EPISODES.items():
        for book, scenario in BOOKS.items():
            part = legs.filter(
                (pl.col("scenario") == scenario)
                & (pl.col("date") > start)
                & (pl.col("date") <= end)
            ).sort("date")
            if part.is_empty():
                raise ValueError(f"No {book} data for {episode}")
            pnl = part.get_column("gross_pnl").to_numpy()
            market = part.get_column("market_return").to_numpy()
            ex_ante = part.get_column("stock_beta").to_numpy() * market
            beta = realized_beta(pnl, market)
            rows.append(
                {
                    "episode": episode,
                    "book": book,
                    "total_pp": 100 * pnl.sum(),
                    "ex_ante_beta": float(
                        part.get_column("stock_beta").to_numpy().mean()
                    ),
                    "ex_ante_beta_pp": 100 * ex_ante.sum(),
                    "ex_ante_residual_pp": 100 * (pnl - ex_ante).sum(),
                    "in_window_beta": beta,
                    "in_window_beta_pp": 100 * beta * market.sum(),
                    "in_window_residual_pp": 100 * (pnl - beta * market).sum(),
                    "market_sum_pp": 100 * float(np.sum(market)),
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    legs = pl.read_parquet(args.input / "scaled_leg_daily.parquet")
    decompose(legs).write_csv(
        args.output / "rally_decomposition.csv", float_precision=4
    )


if __name__ == "__main__":
    main()
