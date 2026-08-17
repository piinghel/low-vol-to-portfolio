# Low-volatility article contribution guide

This repository contains the research code and figures for the low-volatility
portfolio-construction article.

## Working principles

- Keep the signal, portfolio construction, risk diagnostics, and plotting logic
  separate.
- Preserve point-in-time membership, causal feature construction, execution
  timing, return conventions, turnover, and cost assumptions.
- Keep generated files in `output/`; treat them as reproducible outputs rather
  than hand-edited source.
- Keep article figures minimal and free of embedded titles. Use subtle grids
  and consistent colors across related plots.
- Prefer small, explicit changes over speculative abstractions.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run python -m unittest discover -s tests -v
```
