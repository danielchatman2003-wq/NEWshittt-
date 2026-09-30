"""Does BRTI's recent move predict its next few seconds?  (would a forward-looking lean help quotes that land late)

For each (look-back k, look-ahead h): slope of  [p(t+h)-p(t)]  on  [p(t)-p(t-k)], per day, over 14 days.
slope > 0 momentum (lean WITH the move), < 0 mean-reversion, ~0 nothing to lean on.   python research/momentum.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402


def main():
    s0, g = bt.load()
    gf = bt.ffill(g)
    n, day = len(gf), 86400
    print(f"{n / day:.1f} days of 1s BRTI\n")
    print("slope of future move on past move (per $1 of past move, expected future $ move), mean over days +/- std across days")
    print(f"{'past k':>7} |" + "".join(f"  h={h}s      " for h in (1, 2, 3, 5)))
    for k in (1, 2, 5, 10):
        row = []
        for h in (1, 2, 3, 5):
            past = gf[k:n - h] - gf[:n - h - k]
            fut = gf[k + h:] - gf[k:n - h]
            slopes = []
            for d in range(int(n / day)):
                a, b = d * day, min((d + 1) * day, len(past))
                x, y = past[a:b], fut[a:b]
                if len(x) > 1000 and x.var() > 0:
                    slopes.append(np.cov(x, y)[0, 1] / x.var())
            row.append(f"{np.mean(slopes):+.3f}+/-{np.std(slopes):.3f}")
        print(f"{k:>6}s |  " + "   ".join(row))
    # how big is the stale-quote problem in dollars vs a one-cent tick?
    d1 = np.abs(gf[1:] - gf[:-1])
    print(f"\ntypical 1s BRTI move: median ${np.median(d1):.2f}, p90 ${np.percentile(d1, 90):.2f}, p99 ${np.percentile(d1, 99):.2f}")
    print("in probability terms at 30 min left (sigma~6$/sqrt(s)): 1c of fair value ~ $%.0f of BRTI near the money" % (0.01 * 6 * np.sqrt(1760) * np.sqrt(2 * np.pi)))


if __name__ == "__main__":
    main()
