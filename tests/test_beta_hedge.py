"""Causal execution and package accounting for the index overlay."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from low_volatility_factor import beta_hedge


def test_hedge_uses_signal_beta_and_fixed_quantities() -> None:
    dates = [date(2020, 1, d) for d in (3, 6, 7)]
    first, second = date(2020, 1, 1), date(2020, 1, 3)
    daily = pl.DataFrame(
        {
            "date": dates,
            "signal_date": [first, first, second],
            "scenario": ["naive_equal_ls"] * 3,
            "market_return": [0.1, -0.1, 0.2],
        }
    )
    targets = pl.DataFrame(
        {
            "signal_date": [first, second],
            "scenario": ["naive_equal_ls"] * 2,
            "stock_beta": [-1.0, -0.5],
        }
    )
    schedule = pl.DataFrame(
        {
            "signal_date": [first, second],
            "execution_date": [date(2020, 1, 2), date(2020, 1, 6)],
            "effective_return_date": [dates[0], dates[2]],
        }
    )
    result = beta_hedge.hedge_daily(daily, targets, schedule)
    np.testing.assert_allclose(result["hedge_gross_return"], [0.1, -0.11, 0.1])
    np.testing.assert_allclose(result["hedge_turnover"], [1.0, 0.0, 0.49])
    np.testing.assert_allclose(
        result["hedge_cost"], np.array([1.0, 0.0, 0.49]) * 0.0005
    )
    changed = targets.with_columns(
        pl.when(pl.col("signal_date") == second)
        .then(-2.0)
        .otherwise(pl.col("stock_beta"))
        .alias("stock_beta")
    )
    rerun = beta_hedge.hedge_daily(daily, changed, schedule)
    assert result.head(2).equals(rerun.head(2))
    invalid = schedule.with_columns(pl.col("execution_date").alias("signal_date"))
    with pytest.raises(ValueError, match="Signal must precede"):
        beta_hedge.hedge_daily(daily, targets, invalid)
