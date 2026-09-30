"""Is the fair-value math right? Check the three inputs (price to beat K, time left tau, volatility sigma) the only way that
matters: standardise every outcome by the model's own prediction and see if it looks like N(0,1).

  z = (settlement - model_mean) / model_sd       should have std 1 and Normal tails if sigma and the tau-scaling are right.
  std(z) > 1  -> sigma too low (model over-confident, near-resolution bot would be taking hidden risk)
  std(z) < 1  -> sigma too high (model under-confident, leaving trades on the table)
  heavy tails -> Normal is wrong out at 99%, exactly where the near-resolution bot trades.

  python research/vol_check.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import backtest as bt  # noqa: E402

N = 60
LAG, LOOKBACK, PRIOR, WEIGHT = 10, 900, 6.0, 300


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def main():
    s0, g = bt.load()
    gf = bt.ffill(g)
    sig = bt.sigma_series(gf, LAG, LOOKBACK, PRIOR, WEIGHT)
    wins = bt.windows(g, s0)
    print(f"{len(wins)} hourly windows of 1-second BRTI ({len(g) / 86400:.1f} days)\n")
    taus = (1800, 900, 600, 300, 120, 61, 45, 30, 15, 5)
    zs = {t: [] for t in taus}
    probs = []                                              # (model P(Up), outcome) for calibration
    for w in wins:
        te = w["t0"] + bt.HOUR
        for tau in taus:
            now = te - tau
            spot, s = gf[now - s0], sig[now - s0]
            if tau > N:
                mean, var = spot, s * s * (tau - N + N / 3)
            else:
                known = gf[te - N + 1 - s0: now + 1 - s0].sum()
                mean, var = (known + tau * spot) / N, (tau / N) ** 2 * s * s * tau / 3
            sd = max(math.sqrt(var), 0.5)
            zs[tau].append((w["f"] - mean) / sd)
            probs.append((phi((mean - w["k"]) / sd), w["up"]))
    print(f"{'tau':>6} {'n':>4} {'std(z)':>7} {'kurt':>6} {'|z|>2':>7} {'(Normal 4.6%)':>13} {'|z|>2.58':>9} {'(Normal 1.0%)':>13}")
    allz = []
    for tau in taus:
        z = np.array(zs[tau])
        allz += list(z)
        k = ((z - z.mean()) ** 4).mean() / z.var() ** 2
        print(f"{tau:>6} {len(z):>4} {z.std():>7.3f} {k:>6.2f} {np.mean(abs(z) > 2) * 100:>6.1f}% {'':>13} {np.mean(abs(z) > 2.58) * 100:>8.1f}%")
    z = np.array(allz)
    print(f"{'ALL':>6} {len(z):>4} {z.std():>7.3f} {((z - z.mean()) ** 4).mean() / z.var() ** 2:>6.2f} {np.mean(abs(z) > 2) * 100:>6.1f}% "
          f"{'':>13} {np.mean(abs(z) > 2.58) * 100:>8.1f}%")
    print("\ncalibration: when the model says the favourite wins with probability p, how often does it?")
    print(f"{'model says':>12} {'n':>5} {'actual':>8} {'losses':>7}")
    p = np.array([max(a, 1 - a) for a, _ in probs])
    won = np.array([(a >= 0.5) == bool(u) for a, u in probs])
    for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 0.999), (0.999, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"{lo:>5.3f}-{min(hi, 1):<5.3f} {m.sum():>5d} {won[m].mean() * 100:>7.1f}% {int((~won[m]).sum()):>7d}   (model avg {p[m].mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
