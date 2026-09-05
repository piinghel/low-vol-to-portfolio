"""Target sizing can be tested without loading the execution engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from low_volatility_factor import config, portfolio_targets


@pytest.mark.parametrize("names_per_leg", [10, 100])
def test_sizing_caps_and_signed_book_exposures(names_per_leg: int) -> None:
    snapshots = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * (2 * names_per_leg),
            "asset_id_bb_global": [f"stock_{i}" for i in range(2 * names_per_leg)],
            "volatility_decile": [1] * names_per_leg + [10] * names_per_leg,
            "sizing_volatility": [0.10] * names_per_leg + [0.50] * names_per_leg,
            "stock_beta": [0.50] * names_per_leg + [1.50] * names_per_leg,
        }
    )
    targets = portfolio_targets.build_stage_targets(
        snapshots,
        config.DataConfig(data_root=Path(".")),
        config.SignalConfig(),
        config.BucketConfig(),
        config.SizingConfig(),
        config.ScenarioConfig(),
    )
    scaled = targets.lazy().filter(pl.col("scenario") == "vol_scaled_ls").collect()
    exposures = portfolio_targets.summarize_target_exposures(scaled).row(0, named=True)

    # Per-stock cap binds for the 10-name long book; gross cap for 100 names.
    expected_long = min(names_per_leg * 0.04, 1.0)
    assert cast(float, scaled["weight"].abs().max()) <= 0.04
    assert exposures["long_exposure"] == pytest.approx(expected_long)
    assert exposures["short_exposure"] == pytest.approx(0.4)
    assert exposures["stock_beta"] == pytest.approx(expected_long * 0.5 - 0.4 * 1.5)
