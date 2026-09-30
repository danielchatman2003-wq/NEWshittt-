# Polymarket market maker

A two-sided market maker for Polymarket binary (Yes/No) markets. It rests maker-only
limit orders around the book midpoint, skews them against its inventory, and pulls quotes
when the market looks dangerous. **Dry-run by default** - it only trades with `--live`.

## Quick start

```bash
pip install -r requirements.txt
python -m polymarket_mm            # dry-run: reads real books, logs what it WOULD quote
python -m pytest                   # tests
```

Dry-run needs no wallet. To trade for real:

```bash
cp .env.example .env               # fill in PRIVATE_KEY and FUNDER_ADDRESS
python -m polymarket_mm --live
```

Your funder wallet needs USDC on Polygon and the Polymarket exchange allowances set
(placing one order in the Polymarket UI does this for you).

## How it works

1. **Select markets** (Gamma API): active Yes/No markets ranked by 24h volume, filtered to
   mid-priced (0.15-0.85), not resolving within a few days, min volume. Markets paying
   liquidity rewards are ranked first. Re-selected every 30 min.
2. **Quote** every `MM_REFRESH_SECONDS`: `mid = (best bid + best ask) / 2`, then
   `bid/ask = mid - skew*inventory +/- half_spread`, rounded to the tick. No shorting is
   needed - the bid side is `BUY YES` (or `SELL NO` if we hold NO) and the ask side is
   `SELL YES` (or `BUY NO @ 1-ask` if we hold no YES). Inventory is unwound before new
   exposure is opened. See `polymarket_mm/quoter.py`.
3. **Reconcile**: existing orders that still match the desired quote are left alone
   (keeps queue priority); anything else is cancelled and replaced. All orders are
   `post_only`, so they can never cross the spread and pay taker fees.

## Risk controls

| Control | Setting |
|---|---|
| Max shares held per outcome token | `MM_MAX_POSITION` |
| Max USDC committed to resting buys | `MM_MAX_OPEN_NOTIONAL` |
| Pull quotes if mid jumps between cycles | `MM_MOVE_PAUSE` |
| Pull quotes on empty / wide books | `MM_MAX_BOOK_SPREAD` |
| Pull BTC-market quotes on a fast BRTI move or stale BRTI feed (needs Kalshi key) | `MM_BTC_MOVE_PAUSE`, `MM_BTC_WINDOW` |
| Cancel all + exit after repeated API failures | `MM_MAX_ERRORS` |
| Cancel this bot's quotes on Ctrl-C / SIGTERM | always |

## BRTI feed (optional)

Set `KALSHI_API_KEY_ID` / `KALSHI_PRIVATE_KEY_PATH` in `.env` and the bot streams BRTI from Kalshi
(`polymarket_mm/brti.py`). Markets mentioning bitcoin/BTC then pause when BRTI moves more than
`MM_BTC_MOVE_PAUSE` within `MM_BTC_WINDOW` seconds, or if the feed is stale. Check the feed alone with
`python -m polymarket_mm.brti_cli -v`.

## Limitations - read before going live

- **No fill simulation in dry-run**: dry-run shows quotes, not P&L. Start `--live` with
  tiny sizes (`MM_QUOTE_SIZE=5`, `MM_MAX_POSITION=10`) and watch it.
- **Adverse selection is the real risk**: market makers lose when informed traders
  hit stale quotes (news, resolution). The mid-jump pause helps but is not a guarantee.
- **Holding both YES and NO** is not merged back into USDC automatically (needs an
  on-chain `mergePositions` call); it just sits as locked capital worth $1/pair.
- **No exchange heartbeat**: if the process is killed hard (SIGKILL / machine loss),
  resting orders stay on the book until you cancel them in the UI.
- Prediction-market trading may be restricted in your jurisdiction. Check before use.
- This is not financial advice; you can lose the entire deposit.
