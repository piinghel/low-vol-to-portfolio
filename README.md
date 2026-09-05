# Low-volatility factor research

Research code and figures for [Sizing a Low-Volatility
Portfolio](https://piinghel.github.io/quant/2024/12/15/low-volatility-factor.html).

## Run

The backtest uses the author's shared research packages, referenced through
`../../packages/` in `pyproject.toml`. Those packages and a licensed input export
are required for a complete run; the public repository contains the article's
research code and configuration.

```bash
uv sync --dev
uv run low-vol-factor
```

Use `--data-root /path/to/export --output /path/to/run` to select inputs and a
separate run directory. The input export must contain the licensed vendor
price, constituent, and index files; it is not bundled in this repository.

Each run saves the effective configuration, input hashes, sample and quality
checks, metrics, light/dark SVGs, and compact daily P&L for portfolio, books,
and deciles. Those daily Parquet files support reconciliation without repeating
the full feature pipeline. Generated runs remain ignored by Git.

The corrected September 2026 run is retained locally in
`output/turnover-review-2026-09-05/`, including its input hashes and effective
configuration. It values exited holdings at their liquidation prices and counts
each scenario's trades once. Reproduce it with:

```bash
uv run low-vol-factor --data-root /path/to/export --output output/turnover-review-2026-09-05
```

The September 2026 reproduction covers 12 July 1995–27 May 2026. Signals and
P&L use adjusted-price changes; the point-in-time beta diagnostic uses vendor
total returns compounded onto the index calendar. The default beta eligibility
screen leaves complete 126-day ranking histories in all 523,177 retained
stock/rebalance observations. The portfolio is fixed-notional; compounded
performance indices are not financed account histories.

The [September evidence review](docs/evidence-review-2026-09-05.md) reconciles
the corrected table and episodes and records the unresolved terminal-event
treatment. Missing prices carry forward; the backtest does not add separate
delisting or merger payoffs.

## Check

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```
