# Fair value for the BTC hourly Up/Down market - method and evidence

Reproduce: `python research/fetch_brti.py --days 14 && python research/backtest.py && python research/validate.py`

## The model (`polymarket_mm/model.py`)
Up wins iff `avg(last 60 BRTI prints at window end) >= avg(60 prints at window start)` (= `priceToBeat`).
BRTI is modelled as driftless Brownian motion with dollar vol `sigma`/sqrt(s), estimated from 10-second BRTI
changes over the last 15 min, shrunk toward a prior (6 $/sqrt(s)) until there is history. The final average is
Normal with closed-form mean/variance, including the part-realised case inside the last 60s.
`P(Down) = 1 - P(Up)` exactly (unit-tested); Up and Down are scored separately below.

## Evidence (14 days, 333 hourly windows, 1-second BRTI from Kalshi's CF Benchmarks passthrough)
| Check | Result |
|---|---|
| Outcome balance | 175 Up (52.6%) / 158 Down (47.4%) |
| Reproduces Polymarket US's own numbers | 190/190 outcomes agree; reference price error median $0.09, p95 $0.83, max $1.83 |
| Calibration, all p | every bin within +/-2c of realised frequency (n = 38k points) |
| Down-leaning (p<0.5) | predicted Up 0.272 vs realised 0.269 |
| Up-leaning (p>=0.5) | predicted Up 0.737 vs realised 0.749 |
| Brier vs coin-flip | 0.168 vs 0.250 (walk-forward, no lookahead) |
| Tuning (lag, lookback, scale, fat tails) | no held-out improvement over the shipped settings (0.1537 vs 0.1527) -> settings left unchanged |
| Model vs the market's own mid | Brier 0.1677 vs 0.1736; bootstrap CI over windows [-0.019, +0.001] -> **not significantly different** |
| Blending model with market | no gain; pure model scored best |

## What this means
* The fair value is **trustworthy as an anchor** (calibrated both directions, matches exchange settlement).
* It is **not a proven edge over the market**: the market is already about as accurate. Profit has to come from
  spread capture + the maker rebate while avoiding adverse selection, not from out-predicting the market.
* Weakest region: the first ~15 minutes (Brier 0.23, near coin-flip, market slightly better). Consider not
  quoting, or quoting wider, early in the window.

## Not measured
* Fills/PnL of actual quoting (history has mids, not queue position/trades): needs forward paper-trading.
* Regime dependence (14 days, one market regime); fees are taken from the published schedule, not observed.

## Market-making findings from live trading (2026-09-30) and the research behind them
Live result: ~25 fills at 1 contract, roughly -$0.44 on the day; 18 Up buys vs 2 Down buys while BTC fell ~$500.
| Question | Test | Answer |
|---|---|---|
| Is there a trend the model misses? | `research/trend.py`, 1-15 min lookbacks | No: slope ~0.003 (0.3c per sigma), CIs include 0 |
| Does the book lead BRTI (our fair value stale)? | `research/book_lead.py` | No: the book moves TOWARD our fair value (~13% of the gap per minute) |
| Is our model biased against us in a fall? | `research/gap_by_trend.py` | No: gap +0.3c in falling markets (CI includes 0); -1.4c in rising (model lags a little) |
| Are fills worse in a falling market (would a flow filter help)? | `research/flow_filter.py` | No: markout ~-1.3c everywhere, filter changes nothing; simulated Up/Down fills are 50/50 (30.3k/30.2k) |
| Does momentum help at seconds scale? | `research/momentum.py` | No |
| Would directional taker bets pay? | `research/directional.py` | Not significantly; suggestive +2-5c only in the second half of a window |

Conclusion: a passive maker is one-sided only while the market trends one way (bids get hit, the other side has no
sellers). That cannot be quoted away without predicting the trend, which the data says we cannot. What CAN be done is
limit the damage: `DirectionBudget` blocks re-entry in a direction for 10 min after exiting it at a loss and caps
entries to 3 per direction per window. It is a loss limiter, not a source of edge.
