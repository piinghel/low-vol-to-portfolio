# Retired notebook record

`low_vol.ipynb` was retired on 2026-08-16 after the scripted replacement reproduced the research end to end and passed its independent methodology review.

- Original file size: 845,486 bytes
- SHA-256: `6251c6672c0d117c8e7b18757ddede919c733010ae855a7d9b610a8a50f034ff`
- Original modified timestamp: 2026-01-31 15:21:01 CET
- Original structure: 67 cells, including 60 code cells

## Why it was replaced

- It no longer ran against the current shared-library APIs.
- Its return column was future-shifted, making signal timing difficult to audit.
- The article described a $5 filter that the notebook did not apply.
- The article described 60-day sizing and a 4% cap; the notebook used 20 days and 5%.
- Its five buckets had irregular 10/10/60/10/10 widths.
- Its saved “Sharpe” output used a nonstandard geometric-return ratio.
- Transaction-cost logic existed, but the published headline results excluded costs.

## Historical output retained for comparison

The old equal-weight leg volatilities were approximately 12.2% for the low-volatility basket and 38.6% for the high-volatility basket. Its volatility-scaled short exposure was generally 20%–60%, leaving the stock portfolio materially net long. These values are historical context only and are not used by the rewritten article.

The replacement is `uv run low-vol-factor`; its exact assumptions live in `config.py`, and every run writes machine-readable configuration and data manifests.
