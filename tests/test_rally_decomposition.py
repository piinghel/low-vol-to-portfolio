from datetime import date, timedelta

import numpy as np
import polars as pl

from low_volatility_factor.rally_decomposition import decompose


def test_a_book_that_only_carries_beta_has_no_residual():
    days = [date(1998, 10, 8) + timedelta(days=i) for i in range(1, 60)]
    days += [date(2025, 4, 3) + timedelta(days=i) for i in range(1, 60)]
    market = np.random.default_rng(0).normal(0.001, 0.01, len(days))
    rows = []
    for scenario, beta in (("scaled_long_leg", 0.3), ("scaled_short_leg", -0.4)):
        rows += [
            {
                "date": d,
                "scenario": scenario,
                "gross_pnl": beta * m,
                "market_return": m,
                "stock_beta": beta,
            }
            for d, m in zip(days, market, strict=True)
        ]
    result = decompose(pl.DataFrame(rows))
    assert np.allclose(result["ex_ante_residual_pp"], 0.0, atol=1e-10)
    assert np.allclose(result["in_window_residual_pp"], 0.0, atol=1e-10)
    assert np.allclose(result["total_pp"], result["ex_ante_beta_pp"])
