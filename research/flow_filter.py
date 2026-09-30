"""Are passive bids that get filled in a FALLING market worse than in a flat/rising one?  (would a flow filter help?)

Simulation on the market's own mid (Polymarket US price history, 1s carry-forward), minutes 15-56 of each window:
every 5s a passive Up bid rests at mid-d and a passive Down bid (= Up ask) at mid+d. A bid fills when the mid trades
down through it (within 30s), an ask when it trades up through it. Markout = how much the mid moved against us over the
next 60s after the fill (>0 = good for us). Bucketed by the prior-60s mid trend.        python research/flow_filter.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))
import backtest as bt  # noqa: E402
from validate import CACHE, slug_for  # noqa: E402


def main(d=0.015, hold=30, back=60, fwd=60):
    s0, g = bt.load()
    wins = bt.windows(g, s0)
    recs = []     # (side, trend, markout, windowidx)
    n = 0
    for w in wins:
        slug = slug_for(w["t0"])
        fh = CACHE / f"h_{slug}.json"
        if not fh.exists():
            continue
        hist = json.loads(fh.read_text())
        if len(hist) < 50:
            continue
        n += 1
        ts = np.array([h["timestamp"] for h in hist])
        mid = np.array([(h["longPrice"] + 1 - h["shortPrice"]) / 2 for h in hist])
        secs = np.arange(0, bt.HOUR + 1)
        j = np.searchsorted(ts, w["t0"] + secs, side="right") - 1
        M = np.where(j >= 0, mid[np.maximum(j, 0)], np.nan)
        for t in range(900, bt.HOUR - 150 - hold - fwd, 5):
            m0 = M[t]
            if np.isnan(m0) or np.isnan(M[t - back]):
                continue
            trend = m0 - M[t - back]
            B, A = m0 - d, m0 + d
            seg = M[t + 1: t + 1 + hold]
            hit_dn = np.where(seg <= B)[0]
            hit_up = np.where(seg >= A)[0]
            fb = hit_dn[0] if len(hit_dn) else 10 ** 9
            fa = hit_up[0] if len(hit_up) else 10 ** 9
            if fb == fa == 10 ** 9:
                continue
            if fb <= fa:                                   # we BOUGHT Up at B
                s = t + 1 + fb
                recs.append(("buy UP", trend, M[s + fwd] - B, n))
            else:                                          # we SOLD Up (= bought Down) at A
                s = t + 1 + fa
                recs.append(("buy DOWN", trend, A - M[s + fwd], n))
    print(f"{n} windows, {len(recs)} simulated passive fills (spread d={d * 100:.1f}c each side)\n")
    sides = np.array([r[0] for r in recs]); tr = np.array([r[1] for r in recs]); mo = np.array([r[2] for r in recs])
    mo = np.where(np.isnan(mo), 0, mo)
    print(f"{'prior-60s market trend':<26} {'fills Up':>9} {'fills Down':>11} | {'markout Up':>11} {'markout Down':>13}   (markout: >0 good, cents per contract, 60s after fill)")
    for label, lo, hi in (("FALLING  (<= -3c)", -9, -0.03), ("drifting down (-3..-1c)", -0.03, -0.01), ("flat (-1..+1c)", -0.01, 0.01),
                          ("drifting up (+1..+3c)", 0.01, 0.03), ("RISING   (>= +3c)", 0.03, 9)):
        m = (tr > lo) & (tr <= hi) if lo > -9 else (tr <= hi)
        if hi == 9:
            m = tr > lo
        u, dn = m & (sides == "buy UP"), m & (sides == "buy DOWN")
        f = lambda x: "   n/a " if x.sum() < 15 else f"{mo[x].mean() * 100:+6.2f}c"
        print(f"{label:<26} {int(u.sum()):>9} {int(dn.sum()):>11} | {f(u):>11} {f(dn):>13}")
    up, dn = sides == "buy UP", sides == "buy DOWN"
    print(f"\nALL fills: Up {int(up.sum())} / Down {int(dn.sum())}   mean markout Up {mo[up].mean() * 100:+.2f}c  Down {mo[dn].mean() * 100:+.2f}c")
    # the rule under test: don't bid Up into a fall / don't bid Down into a rise
    for theta in (0.02, 0.03, 0.05):
        keep = ~(((sides == "buy UP") & (tr <= -theta)) | ((sides == "buy DOWN") & (tr >= theta)))
        print(f"FILTER theta={theta * 100:.0f}c: fills kept {int(keep.sum())}/{len(keep)} ({keep.mean() * 100:.0f}%), "
              f"Up/Down {int((keep & up).sum())}/{int((keep & dn).sum())}, mean markout {mo[keep].mean() * 100:+.2f}c (was {mo.mean() * 100:+.2f}c)")


if __name__ == "__main__":
    main()
