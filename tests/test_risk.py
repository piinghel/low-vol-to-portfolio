"""Boundary checks for trailing market-beta estimation."""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from low_volatility_factor.config import BetaConfig, DataConfig
from low_volatility_factor.risk import compute_trailing_market_beta


class RiskTests(unittest.TestCase):
    def test_flat_market_produces_null_instead_of_nan_beta(self) -> None:
        dates = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(3)]
        asset_returns = pl.DataFrame(
            {
                "date": dates,
                "asset_id_bb_global": ["A"] * 3,
                "total_return": [0.01, 0.02, 0.03],
            }
        )
        market_returns = pl.DataFrame({"date": dates, "market_return": [0.0, 0.0, 0.0]})

        result = compute_trailing_market_beta(
            asset_returns,
            market_returns,
            DataConfig(data_root=Path(".")),
            BetaConfig(lookback=2, minimum_observations=2),
        ).collect()

        betas = result.get_column("stock_beta")
        self.assertEqual(betas.null_count(), result.height)
        self.assertEqual(betas.is_nan().sum(), 0)


if __name__ == "__main__":
    unittest.main()
