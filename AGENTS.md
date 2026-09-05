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
- Figure 3 must show realized floating exposure. Figure 6 is one 2-by-2 regime
  figure with dot-com on the left, April 2025–May 2026 on the right, and a single
  caption. Do not attribute the later rally to AI without holdings-level
  evidence; do not restore a separate Figure 8.
- Publish matching light/dark layouts for desktop and phone. On phones, stack
  episode groups with growth above contributions; retain the same scales and
  definitions as desktop.
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
