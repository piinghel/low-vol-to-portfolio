import datetime as dt
import tempfile
import unittest
from pathlib import Path

import polars as pl

from low_volatility_factor.config import DataConfig
from low_volatility_factor.data import build_investable_universe


class InvestableUniverseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = DataConfig(data_root=Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_uses_latest_available_complete_snapshot(self):
        prices = pl.DataFrame(
            {
                "date": [
                    dt.date(2024, 1, 15),
                    dt.date(2024, 1, 15),
                    dt.date(2024, 2, 15),
                    dt.date(2024, 2, 15),
                ],
                "asset_id_bb_global": ["A", "B", "A", "C"],
                "px_last": [10.0, 20.0, 11.0, 30.0],
                "px_last_unadjusted": [10.0, 20.0, 11.0, 30.0],
            }
        )
        constituents = pl.DataFrame(
            {
                "date": [
                    dt.date(2024, 1, 1),
                    dt.date(2024, 1, 1),
                    dt.date(2024, 2, 1),
                    dt.date(2024, 2, 1),
                ],
                "asset_id_bb_global": ["A", "B", "B", "C"],
            }
        )

        result = build_investable_universe(prices, constituents, self.config).collect()

        self.assertEqual(
            result.select("date", "asset_id_bb_global").rows(),
            [
                (dt.date(2024, 1, 15), "A"),
                (dt.date(2024, 1, 15), "B"),
                (dt.date(2024, 2, 15), "C"),
            ],
        )

    def test_filters_on_unadjusted_price(self):
        prices = pl.DataFrame(
            {
                "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
                "asset_id_bb_global": ["A", "B"],
                "px_last": [50.0, 50.0],
                "px_last_unadjusted": [4.99, 5.0],
            }
        )
        constituents = pl.DataFrame(
            {
                "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
                "asset_id_bb_global": ["A", "B"],
            }
        )

        result = build_investable_universe(prices, constituents, self.config).collect()

        self.assertEqual(result.get_column("asset_id_bb_global").to_list(), ["B"])


if __name__ == "__main__":
    unittest.main()
