import datetime as dt
import tempfile
import unittest
from pathlib import Path
from typing import cast

import polars as pl

from low_volatility_factor.config import BucketConfig, DataConfig, SignalConfig
from low_volatility_factor.signals import (
    assign_volatility_buckets,
    compute_selection_volatility,
)


def _price_frame(number_of_days: int = 20) -> pl.DataFrame:
    dates = [
        dt.date(2024, 1, 1) + dt.timedelta(days=day) for day in range(number_of_days)
    ]
    return pl.DataFrame(
        {
            "date": dates,
            "asset_id_bb_global": ["A"] * number_of_days,
            "px_last": [
                100.0 * (1.0 + 0.01 * ((day % 4) - 1)) for day in range(number_of_days)
            ],
        }
    )


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_config = DataConfig(data_root=Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_future_prices_do_not_change_past_signal(self):
        signal_config = SignalConfig(windows=(3, 5, 7))
        original = _price_frame()
        changed = original.with_columns(
            pl.when(pl.col("date") > dt.date(2024, 1, 12))
            .then(pl.col("px_last") * 10)
            .otherwise(pl.col("px_last"))
            .alias("px_last")
        )

        original_signal = compute_selection_volatility(
            original, self.data_config, signal_config
        ).collect()
        changed_signal = compute_selection_volatility(
            changed, self.data_config, signal_config
        ).collect()
        cutoff = dt.date(2024, 1, 12)

        self.assertTrue(
            original_signal.filter(pl.col("date") <= cutoff)
            .select("date", signal_config.signal_column)
            .equals(
                changed_signal.filter(pl.col("date") <= cutoff).select(
                    "date", signal_config.signal_column
                )
            )
        )

    def test_deciles_are_balanced_and_ordered(self):
        date = dt.date(2024, 1, 31)
        signals = pl.DataFrame(
            {
                "date": [date] * 100,
                "asset_id_bb_global": [f"A{number:03d}" for number in range(100)],
                "selection_volatility": [number / 100 for number in range(100)],
            }
        )
        signal_config = SignalConfig()
        bucket_config = BucketConfig()

        result = assign_volatility_buckets(
            signals, self.data_config, signal_config, bucket_config
        ).collect()
        counts = result.group_by("volatility_decile").len().sort("volatility_decile")

        self.assertEqual(
            counts.get_column("volatility_decile").to_list(), list(range(1, 11))
        )
        self.assertEqual(counts.get_column("len").to_list(), [10] * 10)
        self.assertLess(
            cast(
                float,
                result.filter(pl.col("volatility_decile") == 1)
                .get_column("selection_volatility")
                .max(),
            ),
            cast(
                float,
                result.filter(pl.col("volatility_decile") == 10)
                .get_column("selection_volatility")
                .min(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
