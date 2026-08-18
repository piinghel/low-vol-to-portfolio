# Low-volatility article contribution guide

This repository contains the research code and figures for the low-volatility
portfolio-construction article.

## Working principles

- Keep the signal, portfolio construction, risk diagnostics, and plotting logic
  separate.
- Use `ml-ranking-tranching-pnl` for holdings-based P&L, execution turnover,
  transaction costs, and floating-weight diagnostics. Keep target construction
  local to this article; do not recreate the package's P&L math here.
- Preserve point-in-time membership, causal feature construction, execution
  timing, return conventions, turnover, and cost assumptions.
- P&L returns use the package's fixed-notional convention. A target quantity is
  fixed between trades; market moves create floating exposures.
- Keep generated files in `output/`; treat them as reproducible outputs rather
  than hand-edited source.
- Keep article figures minimal and free of embedded titles. Use subtle grids
  and consistent colors across related plots.
- Let date-formatted ticks communicate time; omit generic `Date` x-axis labels.
- Figure 4 must show realized floating exposure. Figure 7 is one 2-by-2 regime
  figure with a single caption; do not restore a separate Figure 8.
- Prefer small, explicit changes over speculative abstractions.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```

Keep only tests for point-in-time membership, causal signal timing, sizing
caps, package P&L/turnover/cost identities, and leg-contribution additivity.
