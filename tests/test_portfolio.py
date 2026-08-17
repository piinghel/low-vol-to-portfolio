"""Small hand-calculated checks for the portfolio engine."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl

from low_volatility_factor.backtest import (
    compute_realized_turnover,
    simulate_stock_targets,
)
from low_volatility_factor.config import (
    BucketConfig,
    DataConfig,
    ScenarioConfig,
    SignalConfig,
    SizingConfig,
)
from low_volatility_factor.portfolio import (
    build_stage_targets,
    summarize_target_exposures,
)


class PortfolioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_config = DataConfig(data_root=Path("."))
        self.scenarios = ScenarioConfig()

    def test_scaling_respects_cap_and_does_not_renormalize_short_leg(self) -> None:
        signal_date = date(2024, 1, 2)
        snapshots = pl.DataFrame(
            {
                "date": [signal_date] * 200,
                "asset_id_bb_global": [f"stock_{index}" for index in range(200)],
                "volatility_decile": [1] * 100 + [10] * 100,
                "sizing_volatility": [0.10] * 100 + [0.50] * 100,
                "stock_beta": [0.50] * 100 + [1.50] * 100,
            }
        )
        targets = build_stage_targets(
            snapshots,
            self.data_config,
            SignalConfig(),
            BucketConfig(),
            SizingConfig(),
            self.scenarios,
        )
        scaled = targets.filter(
            pl.col("scenario") == self.scenarios.volatility_scaled_long_short
        )
        exposures = summarize_target_exposures(scaled).row(0, named=True)

        self.assertLessEqual(cast(float, scaled["weight"].abs().max()), 0.04)
        self.assertAlmostEqual(exposures["long_exposure"], 1.0)
        self.assertAlmostEqual(exposures["short_exposure"], 0.4)
        self.assertAlmostEqual(exposures["net_exposure"], 0.6)

        naive = summarize_target_exposures(
            targets.filter(
                pl.col("scenario") == self.scenarios.naive_equal_weight_long_short
            )
        ).row(0, named=True)
        self.assertAlmostEqual(naive["gross_exposure"], 2.0)
        self.assertAlmostEqual(naive["net_exposure"], 0.0)

    def test_fixed_quantity_long_short_pnl(self) -> None:
        signal_date = date(2024, 1, 2)
        return_date = date(2024, 1, 3)
        targets = pl.DataFrame(
            {
                "signal_date": [signal_date, signal_date],
                "asset_id_bb_global": ["long", "short"],
                "scenario": [
                    self.scenarios.naive_equal_weight_long_short,
                    self.scenarios.naive_equal_weight_long_short,
                ],
                "weight": [1.0, -1.0],
                "stock_beta": [0.5, 1.5],
            }
        )
        returns = pl.DataFrame(
            {
                "date": [return_date, return_date],
                "asset_id_bb_global": ["long", "short"],
                "total_return": [0.10, -0.10],
            }
        )
        date_map = pl.DataFrame(
            {
                "date": [return_date],
                "market_return": [0.0],
                "signal_date": [signal_date],
            }
        )

        result = simulate_stock_targets(targets, returns, date_map, self.data_config)
        self.assertAlmostEqual(result["gross_return"][0], 0.20)
        self.assertAlmostEqual(result["portfolio_relative_value"][0], 1.20)
        self.assertAlmostEqual(result["stock_beta"][0], -2.0 / 3.0)

    def test_turnover_uses_drifted_pretrade_weights(self) -> None:
        first_signal = date(2024, 1, 2)
        second_signal = date(2024, 1, 9)
        first_return = date(2024, 1, 3)
        second_return = date(2024, 1, 10)
        targets = pl.DataFrame(
            {
                "signal_date": [
                    first_signal,
                    first_signal,
                    second_signal,
                    second_signal,
                ],
                "asset_id_bb_global": ["a", "b", "a", "b"],
                "scenario": ["test"] * 4,
                "weight": [0.5, 0.5, 0.5, 0.5],
                "stock_beta": [1.0] * 4,
            }
        )
        returns = pl.DataFrame(
            {
                "date": [first_return, first_return, second_return, second_return],
                "asset_id_bb_global": ["a", "b", "a", "b"],
                "total_return": [0.10, -0.10, 0.0, 0.0],
            }
        )
        date_map = pl.DataFrame(
            {
                "date": [first_return, second_return],
                "market_return": [0.0, 0.0],
                "signal_date": [first_signal, second_signal],
            }
        )

        turnover = compute_realized_turnover(
            targets, returns, date_map, self.data_config
        ).sort("signal_date")
        self.assertAlmostEqual(turnover["equity_turnover"][0], 1.0)
        self.assertAlmostEqual(turnover["equity_turnover"][1], 0.10)


if __name__ == "__main__":
    unittest.main()
