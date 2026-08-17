"""Boundary checks for performance metrics."""

from __future__ import annotations

import unittest

import numpy as np
from low_volatility_factor.metrics import _metric_row


class MetricTests(unittest.TestCase):
    def test_rejects_series_that_cannot_produce_valid_metrics(self) -> None:
        invalid_cases = [
            (np.array([0.01]), 252),
            (np.array([0.01, 0.02]), 0),
            (np.array([0.01, -1.0]), 252),
        ]

        for values, annualization in invalid_cases:
            with (
                self.subTest(values=values, annualization=annualization),
                self.assertRaises(ValueError),
            ):
                _metric_row(values, annualization=annualization)


if __name__ == "__main__":
    unittest.main()
