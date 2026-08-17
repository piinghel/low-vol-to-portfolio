# Low-volatility factor research

This standalone project contains the deterministic Polars pipeline and figures
used by the blog post **“The Low-Volatility Factor: Portfolio Construction
Matters.”**

## Run the research

From this project’s root:

```bash
uv sync --dev
uv run low-vol-factor --data-root /path/to/research-data
```

Outputs are written to this project’s `output/latest/` directory. The run records its full configuration, input-file metadata, dependency versions, data-quality checks, targets, daily results, metrics, and figures.

Run the deliberately small correctness suite with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run python -m unittest discover -s tests -v
```

The eleven tests cover point-in-time membership, the $5 unadjusted-price filter, causal signal timing, deterministic deciles, the 4% weight cap and unnormalized short leg, fixed-quantity long/short P&L, turnover against drifted pre-trade weights, and degenerate beta inputs.

## Default specification

- Point-in-time Russell 1000 constituents; minimum unadjusted price of $5.
- Selection signal: mean annualized trailing volatility over 21, 63, and 126 market days.
- Ten equal-count deciles; long decile 1 and short decile 10.
- Weekly signal close, next-market-close execution, subsequent close-to-close return.
- Sizing volatility: 60 market days; 20% stock target; 4% absolute position cap; 100% leg gross cap.
- Beta diagnostic: trailing 252-day covariance estimate, minimum 126 observations.
- No index hedge: the volatility-scaled stock portfolio is already approximately beta-neutral.
- Costs: 5 bps per dollar of equity traded.

Missing held-stock returns are explicitly filled with zero and reported as a position-day rate. A held-return quality gate fails the run above 300% absolute daily return.

## Modules

- `config.py`: research assumptions.
- `data.py`: input loading, calendar alignment, and point-in-time universe.
- `signals.py`: causal volatility features and deciles.
- `risk.py`: trailing stock-beta diagnostic.
- `portfolio.py`: stage targets and exposures.
- `backtest.py`: execution, fixed-quantity P&L, drifted turnover, and costs.
- `metrics.py`: standard performance, exposure, and realized-beta metrics.
- `plots.py`: deterministic article figures, emitted as SVG with PNG fallbacks.
- `run.py`: one-command orchestration and artifacts.
