"""Reconcile the article and remove full-sample realized beta for diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .metrics import _metric_row

RULES = ("naive_equal_ls", "vol_scaled_ls")


def realized_beta(returns: np.ndarray, market: np.ndarray) -> float:
    """OLS slope with an intercept; require complete, nonconstant market data."""
    if returns.size < 2 or not np.isfinite([returns, market]).all():
        raise ValueError("Beta requires at least two complete finite return pairs")
    variance = float(np.var(market, ddof=1))
    if variance <= 0:
        raise ValueError("Market variance must be positive")
    return float(np.cov(returns, market, ddof=1)[0, 1] / variance)


def compare(
    daily: pl.DataFrame, saved_metrics: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    """Keep costs and the intercept; remove only the fitted market component."""
    rows = []
    series = []
    date_market = None
    for rule in RULES:
        part = daily.lazy().filter(pl.col("scenario") == rule).sort("date").collect()
        if part.get_column("date").n_unique() != part.height:
            raise ValueError("Duplicate scenario/date keys")
        pairs = part.select("date", "market_return")
        if date_market is not None and not pairs.equals(date_market):
            raise ValueError("Rules must use identical dates and market returns")
        date_market = pairs
        net = part.get_column("net_return").to_numpy()
        gross = part.get_column("gross_return").to_numpy()
        market = part.get_column("market_return").to_numpy().astype(float)
        turnover = part.get_column("equity_turnover").to_numpy()
        costs = part.get_column("equity_cost").to_numpy()
        np.testing.assert_allclose(gross - net, costs, atol=1e-14, rtol=0)
        np.testing.assert_allclose(costs, 0.0005 * turnover, atol=1e-14, rtol=0)
        original = _metric_row(net, annualization=252)
        saved = (
            saved_metrics.lazy()
            .filter(
                (pl.col("scenario") == rule) & (pl.col("fee_state") == "after_costs")
            )
            .collect()
            .row(0, named=True)
        )
        for key, value in original.items():
            np.testing.assert_allclose(value, saved[key], atol=1e-12, rtol=1e-12)
        beta = realized_beta(net, market)
        adjusted = net - beta * market
        np.testing.assert_allclose(realized_beta(adjusted, market), 0, atol=1e-12)
        for treatment, values in (("original", net), ("beta_removed", adjusted)):
            rows.append(
                {
                    "scenario": rule,
                    "treatment": treatment,
                    **_metric_row(values, annualization=252),
                    "realized_net_beta": realized_beta(values, market),
                    "removed_beta": beta if treatment == "beta_removed" else 0.0,
                    "annual_stock_turnover": float(turnover.sum() * 252 / part.height),
                }
            )
        series.append(
            part.lazy()
            .select(
                "date",
                "scenario",
                "net_return",
                "market_return",
                (
                    pl.col("net_return")
                    - beta * pl.col("market_return").cast(pl.Float64)
                ).alias("beta_removed_return"),
            )
            .collect()
        )
    metrics = pl.DataFrame(rows)
    original_rows = {r["scenario"]: r for r in rows if r["treatment"] == "original"}
    adjusted_rows = {r["scenario"]: r for r in rows if r["treatment"] == "beta_removed"}
    equal, scaled = (original_rows[rule] for rule in RULES)
    market_mean = float(market.mean() * 252)
    raw_gap = float(scaled["arithmetic_return"]) - float(equal["arithmetic_return"])
    beta_component = (
        float(scaled["realized_net_beta"]) - float(equal["realized_net_beta"])
    ) * market_mean
    residual_gap = float(adjusted_rows[RULES[1]]["arithmetic_return"]) - float(
        adjusted_rows[RULES[0]]["arithmetic_return"]
    )
    np.testing.assert_allclose(raw_gap, beta_component + residual_gap, atol=1e-12)
    return (
        metrics,
        pl.concat(series),
        {
            "annual_market_arithmetic_return": market_mean,
            "annual_arithmetic_gap_scaled_minus_equal": raw_gap,
            "beta_component_of_gap": beta_component,
            "remaining_arithmetic_gap": residual_gap,
        },
    )


def episode_betas(daily: pl.DataFrame, legs: pl.DataFrame) -> pl.DataFrame:
    """Signed contribution betas on the full sample and stated close-to-close rallies."""
    frames = pl.concat(
        [
            daily.lazy()
            .filter(pl.col("scenario") == "vol_scaled_ls")
            .select("date", "scenario", "gross_return", "market_return")
            .collect(),
            legs.lazy()
            .select("date", "scenario", "gross_return", "market_return")
            .collect(),
        ]
    )
    rows = []
    for label, start, end in (
        ("full_sample", date(1995, 7, 11), date(2026, 5, 27)),
        ("dot_com_rally", date(1998, 10, 8), date(2000, 3, 9)),
        ("recent_rally", date(2025, 4, 3), date(2026, 5, 27)),
    ):
        for key, part in (
            frames.lazy()
            .filter((pl.col("date") > start) & (pl.col("date") <= end))
            .sort("date")
            .collect()
            .partition_by("scenario", as_dict=True)
            .items()
        ):
            rows.append(
                {
                    "period": label,
                    "scenario": key[0],
                    "observations": part.height,
                    "signed_contribution_beta": realized_beta(
                        part.get_column("gross_return").to_numpy(),
                        part.get_column("market_return").to_numpy().astype(float),
                    ),
                }
            )
    return pl.DataFrame(rows)


def table_html(metrics: pl.DataFrame) -> str:
    """Render the article's original figures plus clearly separated diagnostics."""

    def number(value: float, suffix: str = "%", precision: int = 1) -> str:
        multiplier = 100 if suffix == "%" else 1
        return f"{value * multiplier:.{precision}f}{suffix}".replace("-", "−")

    lines = [
        '<table class="research-table comparison-table portfolio-card-table">',
        (
            "  <caption><strong>Table 1: The sizing comparison, with a check for market exposure.</strong> "
            "12 July 1995–27 May 2026. Returns, volatility and stock turnover are annualized. "
            "Both return columns include the 5 bp stock-trading charge. "
            "The lower rows remove full-sample realized net-return beta, with zero financing and hedge costs; "
            "turnover counts the original stock trades only. Sharpe uses a zero cash rate.</caption>"
        ),
        (
            "  <thead><tr><th>Rule</th><th>Arithmetic return</th><th>Geometric return</th>"
            "<th>Volatility</th><th>Sharpe</th><th>Max drawdown</th><th>Stock turnover</th></tr></thead>"
        ),
        "  <tbody>",
    ]
    for treatment, heading in (
        ("original", "Original portfolios"),
        ("beta_removed", "Realized beta removed · hindsight diagnostic"),
    ):
        lines.append(
            f'    <tr class="period-heading"><th colspan="7">{heading}</th></tr>'
        )
        for rule, label in zip(
            RULES, ("Equal-weight", "Inverse-volatility"), strict=True
        ):
            row = (
                metrics.lazy()
                .filter(
                    (pl.col("scenario") == rule) & (pl.col("treatment") == treatment)
                )
                .collect()
                .row(0, named=True)
            )
            cells = [
                number(row[key])
                for key in ("arithmetic_return", "geometric_return", "volatility")
            ]
            cells += [
                number(row["sharpe_ratio"], "", 2),
                number(row["maximum_drawdown"]),
                number(row["annual_stock_turnover"], "×"),
            ]
            lines.append(
                f'    <tr><th scope="row">{label}</th>'
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>"
            )
    return "\n".join([*lines, "  </tbody>", "</table>", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    paths = [
        args.input / name
        for name in (
            "stage_daily.parquet",
            "stage_metrics.csv",
            "scaled_leg_daily.parquet",
        )
    ]
    daily = pl.scan_parquet(paths[0]).collect()
    metrics, adjusted, decomposition = compare(daily, pl.scan_csv(paths[1]).collect())
    episodes = episode_betas(daily, pl.scan_parquet(paths[2]).collect())
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(args.output / "metrics.csv")
    adjusted.write_parquet(args.output / "daily.parquet")
    episodes.write_csv(args.output / "episode_betas.csv")
    (args.output / "table.html").write_text(table_html(metrics))
    summary = {
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "versions": {"polars": pl.__version__, "numpy": np.__version__},
        "decomposition": decomposition,
        "checks": "Original metrics reconcile; finite matched pairs; unique keys; cost identities; zero adjusted beta; additive arithmetic decomposition.",
        "interpretation": "Full-sample constant-beta diagnostic on existing fixed-notional returns. Zero financing and hedge costs. Residual returns retain differing gross, stock weights and time-varying exposures.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(metrics)
    print(json.dumps(decomposition, indent=2))
    print(episodes)


if __name__ == "__main__":
    main()
