# Low-volatility factor research

Research code and figures for [The Low-Volatility Factor: From Stock Sorts to
Portfolio Risk](https://piinghel.github.io/quant/2024/12/15/low-volatility-factor.html).

## Run

```bash
uv sync --dev
uv run low-vol-factor
```

## Check

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```
