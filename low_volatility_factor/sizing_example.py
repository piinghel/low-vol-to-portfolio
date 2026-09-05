"""Compare sizing rules on invented stock snapshots; no prices or P&L needed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from . import config, portfolio_targets


def main() -> None:
    # Each book has 100 invented stocks. Ranking and beta estimation are outside
    # this example: the snapshots supply their already-calculated inputs.
    snapshots = pl.DataFrame(
        {
            "date": [date(2026, 1, 5)] * 200,
            "asset_id_bb_global": [f"example_{index:03d}" for index in range(200)],
            "volatility_decile": [1] * 100 + [10] * 100,
            "sizing_volatility": [0.10] * 100 + [0.50] * 100,
            "stock_beta": [0.50] * 100 + [1.50] * 100,
        }
    )
    scenarios = config.ScenarioConfig()
    targets = portfolio_targets.build_stage_targets(
        snapshots,
        config.DataConfig(data_root=Path(".")),
        config.SignalConfig(),
        config.BucketConfig(),
        config.SizingConfig(),
        scenarios,
    )
    comparison = (
        portfolio_targets.summarize_target_exposures(targets)
        .lazy()
        .filter(
            pl.col("scenario").is_in(
                [
                    scenarios.naive_equal_weight_long_short,
                    scenarios.volatility_scaled_long_short,
                ]
            )
        )
        .select("scenario", "long_exposure", "short_exposure", "stock_beta")
        .collect()
    )
    print("Synthetic target exposures; short exposure is shown as a positive size.")
    print(comparison)
    print("No realized returns or portfolio-volatility estimates are calculated.")


if __name__ == "__main__":
    main()
