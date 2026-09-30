"""Does the MARKET (book) lead BRTI?   If so, a fair value built from BRTI alone is always a step behind.

Per window, every 5s: gap = market_mid - model_fair.   Then:
   d_fair = fair(t+h) - fair(t)   and   d_mid = mid(t+h) - mid(t)
Slope of d_fair on gap  ~ share of the gap that CLOSES by the model catching up to the market (market leads).
Slope of d_mid  on gap  ~ share of the gap that closes by the market moving to the model (BRTI leads), usually <= 0.
CI: bootstrap over windows.        python research/book_lead.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))
import backtest as bt  # noqa: E402
from validate import CACHE, slug_for  # noqa: E402


def main():
    s0, g = bt.load()
    gf = bt.ffill(g)
    wins = bt.windows(g, s0)
    sig = bt.sigma_series(gf, 10, 900, 6.0, 300)
    HS = (5, 10, 30, 60)
    rows = {h: {"gap": [], "dF": [], "dM": [], "w": []} for h in HS}
    n = 0
    for wi, w in enumerate(wins):
        slug = slug_for(w["t0"])
        fm, fh = CACHE / f"m_{slug}.json", CACHE / f"h_{slug}.json"
        if not (fm.exists() and fh.exists()):
            continue
        m, hist = json.loads(fm.read_text()), json.loads(fh.read_text())
        ptb = ((m.get("assetPriceTerms") or {}).get("priceToBeat") or {}).get("value")
        if not ptb or len(hist) < 50:
            continue
        ts = np.array([h["timestamp"] for h in hist])
        mid = np.array([(h["longPrice"] + 1 - h["shortPrice"]) / 2 for h in hist])
        secs = np.arange(900, bt.HOUR - 200, 5)                       # the period the bot actually quotes
        T = w["t0"] + secs
        j = np.searchsorted(ts, T, side="right") - 1
        ok = j >= 0
        if ok.sum() < 100:
            continue
        n += 1
        idx = T - s0
        sd = sig[idx] * np.sqrt(np.maximum(bt.HOUR - secs - 40, 1))
        fair = stats.norm.cdf((gf[idx] - float(ptb)) / sd)
        M = np.where(ok, mid[np.maximum(j, 0)], np.nan)
        for h in HS:
            k = h // 5
            gap = (M - fair)[:-k]
            dF = fair[k:] - fair[:-k]
            dM = M[k:] - M[:-k]
            v = ~np.isnan(gap) & ~np.isnan(dM)
            rows[h]["gap"].append(gap[v]); rows[h]["dF"].append(dF[v]); rows[h]["dM"].append(dM[v]); rows[h]["w"].append(np.full(v.sum(), n))
    rng = np.random.default_rng(5)
    print(f"{n} windows; sampled every 5s from minute 15 to 56\n")
    print(f"{'horizon':>8} | {'model catches up to market':>30} | {'market moves toward model':>30} | mean |gap|")
    for h in HS:
        gap, dF, dM, W = (np.concatenate(rows[h][k]) for k in ("gap", "dF", "dM", "w"))
        per = [np.where(W == i)[0] for i in range(1, n + 1)]
        def sl(y, idxs): 
            x = gap[idxs]; return np.cov(x, y[idxs])[0, 1] / x.var()
        def ci(y):
            full = sl(y, np.arange(len(gap)))
            bs = [sl(y, np.concatenate([per[i] for i in rng.integers(0, n, n)])) for _ in range(400)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            return f"{full:+.3f} [{lo:+.3f},{hi:+.3f}]"
        print(f"{h:>6} s | {ci(dF):>30} | {ci(dM):>30} | {np.abs(gap).mean() * 100:.1f}c")
    print("\nread: if the LEFT column is clearly positive, the market moves first and our BRTI-only fair value lags it.")


if __name__ == "__main__":
    main()
