"""Would a DIRECTIONAL rule have made money, net of fees?  (buy the side the model says is underpriced, hold to settlement)

Buys at the executable ask (Up ask = longPrice, Down ask = shortPrice), pays the published taker fee
0.0695*p*(1-p), one trade per window (first trigger), walk-forward.   python research/directional.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import backtest as bt  # noqa: E402
from validate import CACHE, slug_for  # noqa: E402

FEE = 0.0695


def asks_at(hist, offs, t0):
    ts = np.array([h["timestamp"] for h in hist])
    up = np.array([h["longPrice"] for h in hist])
    dn = np.array([h["shortPrice"] for h in hist])
    U, D = np.full(len(offs), np.nan), np.full(len(offs), np.nan)
    for i, o in enumerate(offs):
        j = np.searchsorted(ts, t0 + o, side="right") - 1
        if j >= 0:
            U[i], D[i] = up[j], dn[j]
    return U, D


def main():
    s0, g = bt.load()
    gf = bt.ffill(g)
    wins = bt.windows(g, s0)
    sig = bt.sigma_series(gf, 10, 900, 6.0, 300)
    offs = np.arange(15, bt.HOUR - 150 + 1, 30)
    data = []
    for w in wins:
        slug = slug_for(w["t0"])
        fm, fh = CACHE / f"m_{slug}.json", CACHE / f"h_{slug}.json"
        if not (fm.exists() and fh.exists()):
            continue
        m, hist = json.loads(fm.read_text()), json.loads(fh.read_text())
        ptb = ((m.get("assetPriceTerms") or {}).get("priceToBeat") or {}).get("value")
        if not ptb or not hist:
            continue
        idx = w["t0"] - s0 + offs
        sd = sig[idx] * np.sqrt(np.maximum(bt.HOUR - offs - 40, 1))
        p = stats.norm.cdf((gf[idx] - float(ptb)) / sd)
        U, D = asks_at(hist, offs, w["t0"])
        data.append((p, U, D, w["up"]))
    print(f"{len(data)} windows with model + executable market prices\n")
    rng = np.random.default_rng(1)
    print(f"{'min edge':>9} {'trades':>7} {'win%':>6} {'avg cost':>9} {'net P&L/contract':>17}   95% CI (bootstrap over trades)    Up-buys / Down-buys")
    for X in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
        pnl, sides, costs = [], [], []
        for p, U, D, up in data:
            for i in range(len(p)):
                if np.isnan(U[i]) or np.isnan(D[i]) or not (0.03 <= U[i] <= 0.97):
                    continue
                e_up, e_dn = p[i] - U[i], (1 - p[i]) - D[i]
                if max(e_up, e_dn) >= X:
                    side, px = ("U", U[i]) if e_up >= e_dn else ("D", D[i])
                    won = (up == 1) if side == "U" else (up == 0)
                    pnl.append(float(won) - px - FEE * px * (1 - px))
                    sides.append(side); costs.append(px)
                    break                                      # one trade per window
        if len(pnl) < 5:
            print(f"{X:9.2f} {len(pnl):7d}   too few trades")
            continue
        a = np.array(pnl)
        boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(3000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{X:9.2f} {len(a):7d} {np.mean(a + np.array(costs) > 0.5) * 100:5.0f}% {np.mean(costs):9.2f} {a.mean():+17.4f}   [{lo:+.3f}, {hi:+.3f}]   {sides.count('U')} / {sides.count('D')}")


if __name__ == "__main__":
    main()
