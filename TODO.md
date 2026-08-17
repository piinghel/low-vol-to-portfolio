# Low-volatility factor: research plan

The article's central claim is deliberately narrow:

> Equal capital is not equal risk. Scale stock risk first, measure the exposures
> that remain, and add a hedge only when residual beta is material.

## First publishable version — completed 2026-08-16

- [x] Replace the notebook with a one-command Polars package.
- [x] Use strictly trailing adjusted-price returns for the selection signal.
- [x] Change the selection horizons to 21, 63, and 126 market days.
- [x] Replace the irregular 10/10/60/10/10 buckets with ten balanced deciles.
- [x] Use point-in-time Russell 1000 snapshots and a $5 unadjusted-price filter.
- [x] Align stock observations to the index market calendar before computing signals and returns.
- [x] Use weekly signal close → next-close execution → subsequent close-to-close attribution.
- [x] Compare low-vol long, high-vol long, naive 100/100 L/S, and stock-scaled L/S.
- [x] Implement 60-day sizing volatility, a 20% stock target, a 4% position cap, and a 100% leg cap.
- [x] Leave constrained legs underinvested and report long, short, gross, and net exposure explicitly.
- [x] Estimate 252-day stock beta and verify that the scaled stock portfolio is already approximately beta-neutral.
- [x] Measure turnover against drifted pre-trade holdings.
- [x] Apply 5 bps equity trading costs.
- [x] Report arithmetic and geometric return, volatility, standard Sharpe, drawdown, exposure, turnover, ex-ante beta, and realized beta.
- [x] Add a held-return QC gate and report missing-return position-day rates.
- [x] Add eleven focused tests for research-invalidating and boundary failures.
- [x] Regenerate all article figures and machine-readable outputs from one command.
- [x] Rewrite the post for a technical audience and remove introductory volatility/ranking mathematics.
- [x] Build the Jekyll site successfully.
- [x] Complete an independent methodology review, fix every material finding, and pass a follow-up review.
- [x] Retire the notebook while preserving its checksum and historical assumptions.

## Next research pass — prioritized

### Data and implementation realism

- [ ] Model delisting proceeds instead of assigning zero return to missing held-stock observations.
- [ ] Add stock borrow fees, financing, and a liquidity-dependent equity cost model.
- [ ] Investigate the four held high-volatility observations with daily returns above 100%.
- [ ] Add capacity diagnostics using ADV and market capitalization.

### Robustness

- [ ] Compare 10/21/63, 21/63/126, and slower selection horizons.
- [ ] Compare rolling and exponentially weighted selection and sizing volatility.
- [ ] Test alternative rebalance days and biweekly/monthly trading.
- [ ] Add subperiod, crisis-period, and rolling-window results.
- [ ] Add sector exposure and sector-neutral variants.
- [ ] Test alternative beta estimators and define a hedge threshold for cases with material residual beta.
- [ ] Compare against a Russell 1000 benchmark and a published low-volatility index.

### Article maintenance

- [ ] Publish the research code in a versioned repository before linking it from the post.
- [ ] Add a compact reproducibility appendix once the data licensing language is settled.
- [ ] Re-run the pipeline when the source data is extended beyond 2024-10-11.

## Current reproducibility contract

Run from the `ml_rank` root:

```bash
uv sync --extra dev
uv run ruff check projects/blog/low_volatility_factor
uv run ruff format --check projects/blog/low_volatility_factor
uv run ty check projects/blog/low_volatility_factor
uv run low-vol-factor
uv run python -m unittest discover -s projects/blog/low_volatility_factor/tests -v
```

The current defaults and all generated metadata are described in `README.md`, `config.py`, and `output/latest/`.
