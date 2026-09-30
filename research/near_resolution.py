"""Bot #6 'near-resolution' on the 60-min Up/Down: when the model is >= theta sure, buy the winning side at the market's ask and
hold to settlement. Net of the published taker fee. Walk-forward (the model only sees BRTI up to the entry second). One entry per
window (first trigger). The settlement averages the final 60 prints, which the model accounts for inside the last minute.

  python research/near_resolution.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import backtest as bt  # noqa: E402
from polymarket_mm.model import SecondSampler, fair_up  # noqa: E402
from validate import CACHE, slug_for  # noqa: E402

FEE = 0.0695


def build_cands():
    s0, g = bt.load()
    gf = bt.ffill(g)
    wins = bt.windows(g, s0)
    data = []
    for w in wins:
        slug = slug_for(w["t0"])
        fm, fh = CACHE / f"m_{slug}.json", CACHE / f"h_{slug}.json"
        if not (fm.exists() and fh.exists()):
            continue
        m, hist = json.loads(fm.read_text()), json.loads(fh.read_text())
        ptb = ((m.get("assetPriceTerms") or {}).get("priceToBeat") or {}).get("value")
        if not ptb or len(hist) < 50:
            continue
        ts = np.array([h["timestamp"] for h in hist]); up = np.array([h["longPrice"] for h in hist]); dn = np.array([h["shortPrice"] for h in hist])
        data.append((w, float(ptb), ts, up, dn))
    print(f"{len(data)} hourly windows with BRTI + the market's executable asks\n")
    # candidate entries per window: list of (tau, fair, up_ask, dn_ask)
    cands = []
    for w, ptb, ts, up, dn in data:
        smp = SecondSampler(keep=6000)
        t_end = w["t0"] + bt.HOUR
        seen = w["t0"] - 1200
        rows = []
        for t in range(w["t0"] + 600, t_end - 5, 5):                # from minute 10 to 5s before expiry
            for sec in range(seen, t + 1):
                v = gf[sec - s0]
                smp.add(sec, v)
            seen = t + 1
            j = np.searchsorted(ts, t, side="right") - 1
            if j < 0:
                continue
            fair = fair_up(spot=gf[t - s0], k=ptb, now=t, window_end=t_end, sigma=smp.sigma(t), sampler=smp)
            rows.append((t_end - t, fair, up[j], dn[j]))
        cands.append((w["up"], rows))
    return cands


def main():
    cands = build_cands()
    rng = np.random.default_rng(7)
    print(f"{'theta':>6} {'margin':>7} {'trades':>7} {'win%':>6} {'avg cost':>9} {'net/contract':>13}   95% CI              worst  losers")
    for theta in (0.90, 0.95, 0.97, 0.99):
        for margin in (0.01, 0.02):
            pnl, costs, taus = [], [], []
            for up_out, rows in cands:
                for tau, fair, ua, da in rows:
                    if fair >= theta and ua <= 0.995 and fair - ua >= margin:
                        px, won = ua, up_out == 1
                    elif fair <= 1 - theta and da <= 0.995 and (1 - fair) - da >= margin:
                        px, won = da, up_out == 0
                    else:
                        continue
                    pnl.append(float(won) - px - FEE * px * (1 - px)); costs.append(px); taus.append(tau)
                    break
            if len(pnl) < 8:
                print(f"{theta:>6.2f} {margin:>7.2f} {len(pnl):>7d}   too few"); continue
            a = np.array(pnl)
            bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(3000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{theta:>6.2f} {margin:>7.2f} {len(a):>7d} {np.mean(a + np.array(costs) > 0.5) * 100:5.0f}% {np.mean(costs):>9.3f} {a.mean() * 100:>+11.2f}c   [{lo * 100:+.2f}, {hi * 100:+.2f}]c  {a.min() * 100:+6.1f}c {int((a < 0).sum()):>5d}")
    print("\n(each losing trade costs ~the price paid, so ONE reversal erases that many small wins)")
    stop_test(cands)


def stop_test(cands):
    """What if we SELL when the model stops being sure? Exit at the market's bid (1 - the other side's ask) + taker fee."""
    rng = np.random.default_rng(11)
    print("\n== EXIT RULE: sell the moment the model's probability for our side falls to the stop level ==")
    print(f"{'theta':>6} {'stop':>6} {'trades':>7} {'exited':>7} {'win%':>6} {'net/contract':>13}   95% CI           worst    losers")
    for theta in (0.90, 0.95, 0.99):
        for stop in (None, 0.85, 0.75, 0.60):
            pnl, nexit = [], 0
            for up_out, rows in cands:
                for i, (tau, fair, ua, da) in enumerate(rows):
                    if fair >= theta and ua <= 0.995 and fair - ua >= 0.02:
                        side, px = "U", ua
                    elif fair <= 1 - theta and da <= 0.995 and (1 - fair) - da >= 0.02:
                        side, px = "D", da
                    else:
                        continue
                    entry_fee = FEE * px * (1 - px)
                    result = None
                    if stop is not None:
                        for tau2, f2, ua2, da2 in rows[i + 1:]:
                            mine = f2 if side == "U" else 1 - f2
                            if mine <= stop:                                   # the model no longer trusts it: sell into the bid
                                bid = (1 - da2) if side == "U" else (1 - ua2)
                                result = bid - px - entry_fee - FEE * bid * (1 - bid)
                                nexit += 1
                                break
                    if result is None:
                        won = (up_out == 1) if side == "U" else (up_out == 0)
                        result = float(won) - px - entry_fee
                    pnl.append(result)
                    break
            if len(pnl) < 8:
                continue
            a = np.array(pnl)
            bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"{theta:>6.2f} {('none' if stop is None else f'{stop:.2f}'):>6} {len(a):>7d} {nexit:>7d} {np.mean(a > 0) * 100:5.0f}% {a.mean() * 100:>+11.2f}c   [{lo * 100:+.2f}, {hi * 100:+.2f}]c {a.min() * 100:+7.1f}c {int((a < 0).sum()):>7d}")


if __name__ == "__main__":
    main()
