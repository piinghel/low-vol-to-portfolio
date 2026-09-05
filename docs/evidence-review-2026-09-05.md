# September 2026 evidence review

The published sizing comparison uses the local run
`output/turnover-review-2026-09-05/`, generated after turnover correction
`0f8acbe`. The run's configuration and input hashes remain local with its
generated outputs. No licensed source observations are included here.

## Corrected results

The correction values holdings removed at a rebalance using their liquidation
prices and counts trades separately within each scenario. Previously, missing
exit valuations suppressed liquidation turnover, while overlapping scenario
holdings could duplicate the price join. Quantities, gross P&L and the risk
diagnostics were unchanged; additional charged turnover reduced net returns.

| Rule | Net geometric return | Volatility | Sharpe | Maximum drawdown | Annual two-way turnover |
|---|---:|---:|---:|---:|---:|
| Equal weight | −3.3282% | 33.3935% | 0.0663 | −87.8150% | 18.7758× |
| Inverse volatility | 6.7711% | 9.7585% | 0.7203 | −37.9910% | 12.3891× |

These match the article's rounded table and terminal indices of 0.3521 and
7.5411. Reconciliation of saved daily results found a maximum absolute
residual of `4.2e-17` between gross minus net P&L and `0.0005 × turnover`.
The gross sum of the two scaled books matches the scaled portfolio within
`2.1e-17`.

Episode results also match the article. From the close of 8 October 1998 to
9 March 2000, net portfolio return is −37.9910%, the market gains 52.1927%,
and linked before-cost contributions are −10.4103 and −27.1433 percentage
points from longs and shorts. From the close of 3 April 2025 to 27 May 2026,
the corresponding values are −12.6682%, +38.5484%, +4.1858 and −16.2782 points.

Before the presentation refresh below, the published light and dark
performance/drawdown SVGs were byte-identical to this run. The other two published figure pairs had identical visible path
geometry and text to the regenerated figures; timestamps, generated SVG
identifiers and font serialization differ.

## Presentation-only refresh

The subsequent desktop/mobile refresh reads the same saved run; it does not
rerun features or portfolios. Regenerate with:

```sh
POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m low_volatility_factor.article_figures --input output/turnover-review-2026-09-05 --output output/display-review-2026-09-05
```

The three article figures now include light/dark phone layouts. Performance
drawdowns use only a light reference fill; episode panels use common limits
across episodes, visible zero references and a labeled portfolio trough.
These new SVGs are deliberately not byte-identical to the previous exports;
return and contribution calculations are unchanged. The values above remain
the evidence baseline, including the terminal-event limitation below.

## Terminal events remain unresolved

The implemented path is explicit:

1. `simulate_stock_targets` supplies adjusted prices to the shared package's
   quantity-based P&L calculation. The vendor total-return field supports beta
   and data-quality diagnostics; it does not provide portfolio cash flows.
2. Price panels carry the last observed adjusted price forward, both for
   valuation and for the next rebalance's executed notional. Quantities remain
   in the book until the target schedule replaces them. A missing price does
   not itself trigger an exit on the last observed date.
3. Neither this project nor this P&L call adds a delisting loss, cash merger
   payment, share conversion or successor holding. Terminal economics are
   represented only to the extent already embedded in adjusted prices.
4. `prepare_held_returns` trims uncovered dates for a diagnostic. Its output is
   not the book sent to the P&L engine. Its previous docstring incorrectly
   described this trimming as a package delisting convention.

The local export contains merger metadata, but the inspected schema does not
provide a completed-event effective date, cash proceeds per share or exchange
ratio. Announcement date, status, payment type and aggregate deal value do not
determine the payoff of one held share. Those records are not consumed by this
research run. No separate delisting-return input is configured.

A synthetic missing-price test confirms that P&L becomes zero after the last
price, the stale holding remains exposed until the next rebalance, and its
exit is charged at that last price. This verifies the convention, not its
economic accuracy. The held-return quality counts exclude trailing uncovered
dates and therefore do not establish complete terminal-event coverage.

The outstanding task is to match held exits to verified event dates and
security-level terminal payoffs, reconcile those payoffs against existing
adjusted-price changes to avoid double counting, then replay both sizing rules
on the same corrected inputs. This review cannot establish the size or sign
of the missing-event bias. Published return levels remain conditional on this
limitation; the turnover correction does not resolve it.
