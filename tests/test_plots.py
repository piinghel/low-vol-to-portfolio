"""Invariant for the additive leg-contribution chart."""

from datetime import date

import polars as pl
import pytest

from low_volatility_factor.plots import _compound_leg_contribution


def test_leg_contributions_add_to_compounded_combined_wealth() -> None:
    dates = [date(2024, 1, 3), date(2024, 1, 4)]
    combined = pl.DataFrame({"date": dates, "gross_return": [0.10, 0.20]})
    legs = pl.DataFrame(
        {
            "date": dates * 2,
            "scenario": ["long", "long", "short", "short"],
            "portfolio_relative_value": [1.06, 1.12, 1.04, 1.08],
        }
    )

    long = _compound_leg_contribution(combined, legs, "long")
    short = _compound_leg_contribution(combined, legs, "short")
    combined_wealth = (1.0 + combined["gross_return"]).cum_prod()
    reconstructed = long["wealth"] + short["wealth"] - 1.0

    assert reconstructed.to_list() == pytest.approx(combined_wealth.to_list())

    long_from_first_close = _compound_leg_contribution(
        combined,
        legs,
        "long",
        baseline_at_first_date=True,
    )
    short_from_first_close = _compound_leg_contribution(
        combined,
        legs,
        "short",
        baseline_at_first_date=True,
    )
    reconstructed_from_first_close = (
        long_from_first_close["wealth"] + short_from_first_close["wealth"] - 1.0
    )

    assert reconstructed_from_first_close.to_list() == pytest.approx([1.0, 1.2])
