from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from low_volatility_factor.plots import _compound_leg_contribution


def test_leg_contributions_add_to_combined_gross_wealth() -> None:
    combined = pl.DataFrame(
        {
            "date": [
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
            ],
            "signal_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 4),
                date(2024, 1, 4),
            ],
            "gross_return": [0.10, 1.2 / 1.1 - 1.0, 0.05, 1.32 / 1.26 - 1.0],
        }
    )
    legs = pl.DataFrame(
        {
            "date": [
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
                date(2024, 1, 8),
            ]
            * 2,
            "signal_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 4),
                date(2024, 1, 4),
            ]
            * 2,
            "scenario": ["long"] * 4 + ["short"] * 4,
            "portfolio_relative_value": [
                1.06,
                1.10,
                1.02,
                1.05,
                1.04,
                1.10,
                1.03,
                1.05,
            ],
        }
    )

    long = _compound_leg_contribution(combined, legs, "long")
    short = _compound_leg_contribution(combined, legs, "short")
    combined_wealth = (1.0 + combined["gross_return"]).cum_prod()
    reconstructed = long["wealth"] + short["wealth"] - 1.0

    assert reconstructed.to_list() == pytest.approx(combined_wealth.to_list())
