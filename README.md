# Low-volatility factor research

Research code and figures for [Sizing a Low-Volatility
Portfolio](https://piinghel.github.io/quant/2024/12/15/low-volatility-factor.html).

## Start with the sizing rule

From the repository root, with Python 3.12 or later and
[uv](https://docs.astral.sh/uv/):

```bash
uv run --isolated --no-project --python 3.12 --with polars==1.44.1 python -m low_volatility_factor.sizing_example
```

This uses the actual target-construction code on invented stock snapshots.
Equal weights give each book 1.0 notional. Volatility scaling leaves the
low-volatility long book at 1.0 and reduces the high-volatility short book to
0.4. It does not calculate returns or estimate realized portfolio volatility.
The command needs no input files and does not install the full backtest stack.

The method is split into a few direct steps:

| Step | Source |
| --- | --- |
| Trailing volatility, ranking and beta | [features.py](low_volatility_factor/features.py) |
| Stock selection and target sizing | [portfolio_targets.py](low_volatility_factor/portfolio_targets.py) |
| Timing and execution-engine calls | [backtest.py](low_volatility_factor/backtest.py) |
| Experiment settings and orchestration | [config.py](low_volatility_factor/config.py), [run.py](low_volatility_factor/run.py) |

Run the signal and sizing tests without the backtest dependencies:

```bash
uv run --isolated --no-project --python 3.12 --with polars==1.44.1 --with pytest python -m pytest tests/test_features.py tests/test_targets.py -q
```

## Full backtest

The full runner additionally requires the shared packages referenced in
`pyproject.toml` and price, point-in-time constituent, and index inputs. These
are not bundled in this checkout; the synthetic example above runs independently.

```bash
uv sync --dev
uv run low-vol-factor
```

Use `--data-root /path/to/export --output /path/to/run` to select inputs and a
separate run directory. Column names and file patterns are specified in
`DataConfig` in [config.py](low_volatility_factor/config.py).

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
