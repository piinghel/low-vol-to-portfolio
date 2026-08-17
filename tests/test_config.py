"""Validation checks for research configuration boundaries."""

from __future__ import annotations

import unittest
from pathlib import Path

from low_volatility_factor.config import (
    BacktestConfig,
    BetaConfig,
    CostConfig,
    DataConfig,
    ResearchConfig,
    ScenarioConfig,
    SignalConfig,
    SizingConfig,
)


class ConfigTests(unittest.TestCase):
    def test_rejects_invalid_component_values(self) -> None:
        invalid_factories = [
            lambda: SignalConfig(minimum_annualized_volatility=0),
            lambda: SizingConfig(volatility_window=1),
            lambda: SizingConfig(annualized_stock_volatility_target=0),
            lambda: SizingConfig(maximum_absolute_stock_weight=1.1),
            lambda: SizingConfig(maximum_leg_gross_exposure=0),
            lambda: CostConfig(equity_cost_bps=-0.1),
            lambda: BetaConfig(minimum_observations=253),
            lambda: BetaConfig(beta_clip=(1.0, 1.0)),
            lambda: BacktestConfig(annualization_factor=0),
            lambda: ScenarioConfig(high_volatility_long="low_vol_long"),
        ]

        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises(ValueError):
                factory()

    def test_requires_one_annualization_convention(self) -> None:
        with self.assertRaisesRegex(ValueError, "annualization factors must match"):
            ResearchConfig(
                data=DataConfig(data_root=Path(".")),
                signal=SignalConfig(annualization_factor=252),
                backtest=BacktestConfig(annualization_factor=260),
            )


if __name__ == "__main__":
    unittest.main()
