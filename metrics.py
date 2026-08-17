"""Standard performance metrics and time-series transformations."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .frames import require_finite_float


def _metric_row(
    values: np.ndarray,
    *,
    annualization: int,
) -> dict[str, float | int]:
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        raise ValueError("At least two finite returns are required")
    if np.any(clean <= -1.0):
        raise ValueError("Returns must be greater than -100% to compound wealth")
    arithmetic_return = float(clean.mean() * annualization)
    volatility = float(clean.std(ddof=1) * math.sqrt(annualization))
    wealth = np.cumprod(1.0 + clean)
    years = clean.size / annualization
    geometric_return = float(wealth[-1] ** (1.0 / years) - 1.0)
    wealth_with_initial_nav = np.concatenate(([1.0], wealth))
    running_max = np.maximum.accumulate(wealth_with_initial_nav)
    drawdown = wealth_with_initial_nav / running_max - 1.0
    return {
        "observations": int(clean.size),
        "arithmetic_return": arithmetic_return,
        "geometric_return": geometric_return,
        "volatility": volatility,
        "sharpe_ratio": arithmetic_return / volatility
        if volatility > 0
        else float("nan"),
        "maximum_drawdown": float(drawdown.min()),
        "terminal_wealth": float(wealth[-1]),
    }


def compute_stage_metrics(
    daily: pl.DataFrame,
    *,
    annualization: int = 252,
) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario in daily.get_column("scenario").unique().sort().to_list():
        part = daily.filter(pl.col("scenario") == scenario).sort("date")
        years = part.height / annualization
        market = part.get_column("market_return").to_numpy()
        gross_returns = part.get_column("gross_return").to_numpy()
        finite_pair = np.isfinite(market) & np.isfinite(gross_returns)
        market_variance = float(np.var(market[finite_pair], ddof=1))
        realized_beta = (
            float(np.cov(gross_returns[finite_pair], market[finite_pair], ddof=1)[0, 1])
            / market_variance
            if market_variance > 0
            else float("nan")
        )
        exposure_stats = {
            "average_gross_exposure": require_finite_float(
                part.get_column("gross_exposure").mean(), "average gross exposure"
            ),
            "average_net_exposure": require_finite_float(
                part.get_column("net_exposure").mean(), "average net exposure"
            ),
            "average_long_exposure": require_finite_float(
                part.get_column("long_exposure").mean(), "average long exposure"
            ),
            "average_short_exposure": require_finite_float(
                part.get_column("short_exposure").mean(), "average short exposure"
            ),
            "average_ex_ante_stock_beta": require_finite_float(
                part.get_column("stock_beta").mean(), "average ex-ante stock beta"
            ),
            "average_absolute_ex_ante_stock_beta": require_finite_float(
                part.get_column("stock_beta").abs().mean(),
                "average absolute ex-ante stock beta",
            ),
            "realized_market_beta": realized_beta,
            "annualized_equity_turnover": require_finite_float(
                part.get_column("equity_turnover").sum(), "total equity turnover"
            ),
        }
        exposure_stats["annualized_equity_turnover"] /= years
        gross = _metric_row(
            part.get_column("gross_return").to_numpy(), annualization=annualization
        )
        net = _metric_row(
            part.get_column("net_return").to_numpy(), annualization=annualization
        )
        rows.append(
            {
                "scenario": scenario,
                "fee_state": "before_costs",
                **gross,
                **exposure_stats,
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "fee_state": "after_costs",
                **net,
                **exposure_stats,
            }
        )
    return pl.DataFrame(rows).sort("scenario", "fee_state")


def compute_simple_metrics(
    daily: pl.DataFrame,
    *,
    return_column: str = "gross_return",
    annualization: int = 252,
) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario in daily.get_column("scenario").unique().sort().to_list():
        part = daily.filter(pl.col("scenario") == scenario).sort("date")
        rows.append(
            {
                "scenario": scenario,
                **_metric_row(
                    part.get_column(return_column).to_numpy(),
                    annualization=annualization,
                ),
            }
        )
    return pl.DataFrame(rows).sort("scenario")


def cumulative_returns(
    daily: pl.DataFrame,
    *,
    return_column: str,
) -> pl.DataFrame:
    return (
        daily.sort("scenario", "date")
        .with_columns(
            (1.0 + pl.col(return_column))
            .cum_prod()
            .over("scenario", order_by="date")
            .alias("wealth")
        )
        .select("date", "scenario", "wealth")
    )


def drawdown_series(
    daily: pl.DataFrame,
    *,
    return_column: str,
) -> pl.DataFrame:
    wealth = cumulative_returns(daily, return_column=return_column)
    return wealth.with_columns(
        (
            pl.col("wealth")
            / pl.col("wealth")
            .cum_max()
            .over("scenario", order_by="date")
            .clip(lower_bound=1.0)
            - 1.0
        ).alias("drawdown")
    )
