"""Shared dataframe boundary helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl

Frame = pl.DataFrame | pl.LazyFrame


def as_lazy(frame: Frame) -> pl.LazyFrame:
    """Return a lazy view without collecting an existing lazy frame."""

    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def require_columns(frame: pl.LazyFrame, columns: Sequence[str], name: str) -> None:
    """Reject a dataframe that does not satisfy a stage's column contract."""

    available = set(frame.collect_schema().names())
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_finite_float(value: object, name: str) -> float:
    """Return a numeric dataframe scalar or reject an invalid aggregate result."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric scalar, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result
