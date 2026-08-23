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
- P&L uses quantities that stay fixed between trades. The package's
  fixed-notional mode refers to the return denominator; market moves create
  floating stock exposures, which are used for the diagnostics and figures.
- Keep generated files in `output/`; treat them as reproducible outputs rather
  than hand-edited source.
- Keep article figures minimal and free of embedded titles. Use subtle grids
  and consistent colors across related plots.
- Let date-formatted ticks communicate time; omit generic `Date` x-axis labels.
- Figure 4 must show realized floating exposure. Figure 7 is one 2-by-2 regime
  figure with dot-com on the left, the AI rally on the right, and a single
  caption; do not restore a separate Figure 8.
- Publish desktop/mobile and light/dark figures as SVGs only.
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
