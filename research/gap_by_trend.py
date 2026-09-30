"""Is our model systematically ABOVE the book when the market is falling (and below when rising)?
If so, centring quotes on the model makes Up bids sit at the touch and Down bids cents behind in a fall: one-sided.
gap = model_fair - market_mid, bucketed by the prior-60s market trend.     python research/gap_by_trend.py
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
    T, G, W = [], [], []
    n = 0
    for w in wins:
        slug = slug_for(w["t0"])
        fm, fh = CACHE / f"m_{slug}.json", CACHE / f"h_{slug}.json"
        if not (fm.exists() and fh.exists()):
            continue
        m, hist = json.loads(fm.read_text()), json.loads(fh.read_text())
        ptb = ((m.get("assetPriceTerms") or {}).get("priceToBeat") or {}).get("value")
        if not ptb or len(hist) < 50:
            continue
        n += 1
        ts = np.array([h["timestamp"] for h in hist]); mid = np.array([(h["longPrice"] + 1 - h["shortPrice"]) / 2 for h in hist])
        secs = np.arange(900, bt.HOUR - 200, 5)
        j = np.searchsorted(ts, w["t0"] + secs, side="right") - 1
        jb = np.searchsorted(ts, w["t0"] + secs - 60, side="right") - 1
        ok = (j >= 0) & (jb >= 0)
        idx = w["t0"] + secs - s0
        fair = stats.norm.cdf((gf[idx] - float(ptb)) / (sig[idx] * np.sqrt(np.maximum(bt.HOUR - secs - 40, 1))))
        M = mid[np.maximum(j, 0)]
        T.append((M - mid[np.maximum(jb, 0)])[ok]); G.append((fair - M)[ok]); W.append(np.full(ok.sum(), n))
    T, G, W = map(np.concatenate, (T, G, W))
    rng = np.random.default_rng(2)
    print(f"{n} windows, {len(T)} samples.  gap = model - market (cents).  + means the model is MORE BULLISH on Up than the book\n")
    print(f"{'prior-60s market trend':<26} {'n':>7} {'mean gap':>10}   95% CI (over windows)")
    for label, lo, hi in (("FALLING  (<= -3c)", -9, -0.03), ("drifting down (-3..-1c)", -0.03, -0.01), ("flat (-1..+1c)", -0.01, 0.01),
                          ("drifting up (+1..+3c)", 0.01, 0.03), ("RISING   (>= +3c)", 0.03, 9)):
        m = (T <= hi) if lo == -9 else (T > lo) if hi == 9 else ((T > lo) & (T <= hi))
        per = [G[(W == i) & m].mean() for i in range(1, n + 1) if ((W == i) & m).sum() > 3]
        per = np.array(per)
        bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(1500)]
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        print(f"{label:<26} {int(m.sum()):>7} {G[m].mean() * 100:>+9.2f}c   [{lo_ * 100:+.2f}, {hi_ * 100:+.2f}]c")
    print("\nIf 'FALLING' is clearly positive and 'RISING' clearly negative, centring on the model makes us one-sided with the trend.")


if __name__ == "__main__":
    main()
