"""High-value portfolio-construction and package-PnL invariants."""

from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from low_volatility_factor.backtest import prepare_held_returns, simulate_stock_targets
from low_volatility_factor.config import (
    BucketConfig,
    CostConfig,
    DataConfig,
    ScenarioConfig,
    SignalConfig,
    SizingConfig,
)
from low_volatility_factor.portfolio_targets import (
    build_stage_targets,
    summarize_target_exposures,
)


def test_scaling_respects_cap_and_keeps_short_leg_underinvested() -> None:
    snapshots = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 200,
            "asset_id_bb_global": [f"stock_{index}" for index in range(200)],
            "volatility_decile": [1] * 100 + [10] * 100,
            "sizing_volatility": [0.10] * 100 + [0.50] * 100,
            "stock_beta": [0.50] * 100 + [1.50] * 100,
        }
    )
    targets = build_stage_targets(
        snapshots,
        DataConfig(data_root=Path(".")),
        SignalConfig(),
        BucketConfig(),
        SizingConfig(),
        ScenarioConfig(),
    )
    scaled = targets.filter(pl.col("scenario") == "vol_scaled_ls")
    exposures = summarize_target_exposures(scaled).row(0, named=True)

    assert cast(float, scaled["weight"].abs().max()) <= 0.04
    assert exposures["long_exposure"] == 1.0
    assert exposures["short_exposure"] == 0.4


def _fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    targets = pl.DataFrame(
        {
            "signal_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "asset_id_bb_global": ["long", "short"],
            "scenario": ["test", "test"],
            "leg": ["long", "short"],
            "weight": [1.0, -1.0],
            "stock_beta": [0.5, 1.5],
        }
    )
    prices = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 2 + [date(2024, 1, 3)] * 2,
            "asset_id_bb_global": ["long", "short"] * 2,
            "px_last": [100.0, 100.0, 110.0, 90.0],
            "total_return": [0.0, 0.0, 0.10, -0.10],
        }
    )
    date_to_signal = pl.DataFrame(
        {
            "date": [date(2024, 1, 3)],
            "signal_date": [date(2024, 1, 2)],
            "market_return": [0.0],
        }
    )
    schedule = pl.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)],
            "execution_date": [date(2024, 1, 2)],
            "effective_return_date": [date(2024, 1, 3)],
        }
    )
    return targets, prices, date_to_signal, schedule


def test_package_pnl_uses_fixed_notional_and_floating_weights() -> None:
    targets, prices, date_to_signal, schedule = _fixture()
    result = simulate_stock_targets(
        targets,
        prices,
        date_to_signal,
        schedule,
        DataConfig(data_root=Path(".")),
        CostConfig(equity_cost_bps=5.0),
    ).row(0, named=True)

    assert result["gross_return"] == 0.20
    assert result["net_return"] == 0.199
    assert result["equity_turnover"] == 2.0
    assert result["long_exposure"] == 1.10
    assert result["short_exposure"] == 0.90


@pytest.mark.parametrize("scenario_names", [["one"], ["one", "two"]])
@pytest.mark.parametrize("next_asset, expected_turnover", [("A", 0.1), ("B", 2.1)])
def test_rebalance_costs_use_drifted_holdings_for_each_scenario(
    scenario_names: list[str], next_asset: str, expected_turnover: float
) -> None:
    dates = [date(2024, 1, day) for day in (2, 3, 4)]
    targets = pl.DataFrame(
        [
            {
                "signal_date": signal_date,
                "asset_id_bb_global": asset,
                "scenario": scenario,
                "weight": 1.0,
                "stock_beta": 1.0,
            }
            for scenario in scenario_names
            for signal_date, asset in zip(dates[:2], ["A", next_asset], strict=True)
        ]
    )
    prices = pl.DataFrame(
        {
            "date": [day for day in dates for _ in range(2)],
            "asset_id_bb_global": ["A", "B"] * 3,
            "px_last": [100.0, 100.0, 110.0, 100.0, 121.0, 110.0],
        }
    )
    date_to_signal = pl.DataFrame(
        {"date": dates[1:], "signal_date": dates[:2], "market_return": [0.0, 0.0]}
    )
    schedule = pl.DataFrame(
        {
            "signal_date": dates[:2],
            "execution_date": dates[:2],
            "effective_return_date": dates[1:],
        }
    )
    result = simulate_stock_targets(
        targets,
        prices,
        date_to_signal,
        schedule,
        DataConfig(data_root=Path(".")),
        CostConfig(equity_cost_bps=5.0),
    )
    for scenario in scenario_names:
        daily = (
            result.lazy().filter(pl.col("scenario") == scenario).sort("date").collect()
        )
        assert daily["gross_return"].to_list() == pytest.approx([0.1, 0.1])
        assert daily["equity_turnover"].to_list() == pytest.approx(
            [1.0, expected_turnover]
        )
        assert daily["equity_cost"].to_list() == pytest.approx(
            [0.0005, expected_turnover * 0.0005]
        )


def test_held_return_quality_marks_internal_missing_dates() -> None:
    targets = pl.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)],
            "asset_id_bb_global": ["asset"],
            "scenario": ["test"],
            "weight": [1.0],
        }
    )
    asset_returns = pl.DataFrame(
        {
            "date": [date(2024, 1, 3), date(2024, 1, 5)],
            "asset_id_bb_global": ["asset", "asset"],
            "total_return": [0.10, 0.20],
        }
    )
    date_to_signal = pl.DataFrame(
        {
            "date": [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
            "signal_date": [date(2024, 1, 2)] * 3,
        }
    )

    result = prepare_held_returns(
        targets,
        asset_returns,
        date_to_signal,
        DataConfig(data_root=Path(".")),
    )

    assert result.get_column("missing_return").to_list() == [False, True, False]
    assert result.get_column("total_return").to_list() == [0.10, 0.0, 0.20]


def test_missing_terminal_price_carries_until_rebalance_and_charges_exit() -> None:
    """Characterize stale-price handling; this is not an event-payoff model."""
    dates = [date(2024, 1, day) for day in (2, 3, 4, 5)]
    targets = pl.DataFrame(
        {
            "signal_date": [dates[0], dates[2]],
            "asset_id_bb_global": ["A", "B"],
            "scenario": ["test", "test"],
            "weight": [1.0, 1.0],
            "stock_beta": [1.0, 1.0],
        }
    )
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1], *dates],
            "asset_id_bb_global": ["A", "A", "B", "B", "B", "B"],
            "px_last": [100.0, 80.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    date_to_signal = pl.DataFrame(
        {
            "date": dates[1:],
            "signal_date": [dates[0], dates[0], dates[2]],
            "market_return": [0.0, 0.0, 0.0],
        }
    )
    schedule = pl.DataFrame(
        {
            "signal_date": [dates[0], dates[2]],
            "execution_date": [dates[0], dates[2]],
            "effective_return_date": [dates[1], dates[3]],
        }
    )
    result = simulate_stock_targets(
        targets,
        prices,
        date_to_signal,
        schedule,
        DataConfig(data_root=Path(".")),
        CostConfig(equity_cost_bps=5.0),
    )

    assert result["gross_return"].to_list() == pytest.approx([-0.2, 0.0, 0.0])
    assert result["gross_exposure"].to_list() == pytest.approx([0.8, 0.8, 1.0])
    assert result["equity_turnover"].to_list() == pytest.approx([1.0, 0.0, 1.8])
    assert result["equity_cost"].to_list() == pytest.approx([0.0005, 0.0, 0.0009])
