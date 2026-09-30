"""Backtest the hourly BTC Up/Down fair-value model on historical BRTI (both Up AND Down outcomes).

  python research/fetch_brti.py --days 14      # once
  python research/backtest.py                  # full report

Everything is walk-forward: the model only sees prints strictly before each evaluation time.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from polymarket_mm.model import SecondSampler, fair_up  # noqa: E402

DATA = ROOT / "data" / "brti"
HOUR = 3600
FLOOR = 2.0


# ---------- data ----------------------------------------------------------------
def load():
    parts = [np.load(f) for f in sorted(DATA.glob("*.npy")) if len(np.load(f))]
    big = np.concatenate(parts)
    secs, px = big[:, 0].astype(np.int64), big[:, 1]
    s0 = int(secs.min())
    g = np.full(int(secs.max()) - s0 + 1, np.nan)
    g[secs - s0] = px
    return s0, g


def ffill(g):
    idx = np.where(~np.isnan(g), np.arange(len(g)), 0)
    np.maximum.accumulate(idx, out=idx)
    return g[idx]


def window_avg(g, s0, end, n=60):
    """Mean of the n one-second prints ending at `end` inclusive (needs >=80% present), else nan."""
    a = g[end - n + 1 - s0: end + 1 - s0]
    if len(a) < n or np.isnan(a).sum() > 0.2 * n:
        return np.nan
    return float(np.nanmean(a))


def windows(g, s0):
    first = (s0 // HOUR + 2) * HOUR
    out = []
    for t0 in range(first, s0 + len(g) - HOUR, HOUR):
        k, f = window_avg(g, s0, t0), window_avg(g, s0, t0 + HOUR)
        if not (np.isnan(k) or np.isnan(f)):
            out.append({"t0": t0, "k": k, "f": f, "up": int(round(f, 2) >= round(k, 2))})
    return out


# ---------- vectorised model (must equal polymarket_mm.model; asserted below) ----------
def sigma_series(gf, lag, lookback, prior, weight, scale=1.0):
    n = len(gf)
    d = np.full(n, np.nan)
    d[lag:] = gf[lag:] - gf[:-lag]
    sq = np.nan_to_num(d ** 2)
    valid = (~np.isnan(d)).astype(float)
    cs, cv = np.concatenate([[0], np.cumsum(sq)]), np.concatenate([[0], np.cumsum(valid)])
    lo = np.maximum(np.arange(n) - lookback, 0)
    hi = np.arange(n) + 1
    cnt = cv[hi] - cv[lo]
    ss = cs[hi] - cs[lo]
    obs = np.sqrt(np.where(cnt > 0, ss / np.maximum(cnt, 1), 0) / lag)
    w = cnt / (cnt + weight)
    return np.maximum(FLOOR, w * obs + (1 - w) * prior) * scale


def prob(z, dist):
    if dist == "normal":
        return stats.norm.cdf(z)
    nu = dist  # student-t with variance matched to the normal
    return stats.t.cdf(z * np.sqrt(nu / (nu - 2)), nu)


def evaluate(gf, s0, wins, sigma, dist="normal", step=30, t_lo=15, t_hi=HOUR - 150):
    """Rows: window idx, seconds-in, tau, p_up, outcome."""
    rows = []
    offs = np.arange(t_lo, t_hi + 1, step)
    for i, w in enumerate(wins):
        idx = w["t0"] - s0 + offs
        spot = gf[idx]
        tau = HOUR - offs
        sd = sigma[idx] * np.sqrt(np.maximum(tau - 60 + 20, 1))
        p = prob((spot - w["k"]) / sd, dist)
        rows.append(np.column_stack([np.full(len(offs), i), offs, tau, p, np.full(len(offs), w["up"])]))
    return np.vstack(rows)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib(p, y, bins=(0, .05, .15, .3, .5, .7, .85, .95, 1.0001)):
    lines = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            lines.append(f"    p in [{lo:.2f},{min(hi, 1):.2f})  n={int(m.sum()):6d}  predicted={p[m].mean():.3f}  realised Up={y[m].mean():.3f}  gap={y[m].mean() - p[m].mean():+.3f}")
    return "\n".join(lines)


def main():
    s0, g = load()
    gf = ffill(g)
    wins = windows(g, s0)
    ups = sum(w["up"] for w in wins)
    print(f"data: {len(g) / 86400:.1f} days of 1s BRTI, {np.isnan(g).mean() * 100:.2f}% missing prints")
    print(f"windows: {len(wins)}   settled UP {ups} ({ups / len(wins):.1%})   settled DOWN {len(wins) - ups} ({1 - ups / len(wins):.1%})\n")

    # equivalence with production code
    samp = SecondSampler(keep=5000)
    sig_v = sigma_series(gf, 10, 900, 6.0, 300)
    w = wins[len(wins) // 2]
    for s in range(w["t0"] - 1000, w["t0"] + 1500):
        if not np.isnan(g[s - s0]):
            samp.add(s, g[s - s0])
    t = w["t0"] + 1500
    fair_prod = fair_up(spot=samp.last(), k=w["k"], now=t, window_end=w["t0"] + HOUR, sigma=samp.sigma(t), sampler=samp)
    sd = sig_v[t - s0] * np.sqrt(HOUR - 1500 - 40)
    fair_vec = float(stats.norm.cdf((gf[t - s0] - w["k"]) / sd))
    print(f"vectorised vs production model at one point: {fair_vec:.4f} vs {fair_prod:.4f}")
    assert abs(fair_vec - fair_prod) < 0.01, "backtest math diverges from production model"

    split = int(len(wins) * 0.7)
    train, test = wins[:split], wins[split:]
    print(f"walk-forward split: tune on first {len(train)} windows, report on last {len(test)} (never tuned on)\n")

    # ---- 1. current production settings, all data, by side
    prod = evaluate(gf, s0, wins, sig_v)
    p, y = prod[:, 3], prod[:, 4]
    print("== CURRENT model (lag10, lookback900, prior6, normal), all windows ==")
    print(f"  Brier {brier(p, y):.4f} (coin-flip 0.2500)   logloss {logloss(p, y):.4f} (coin-flip 0.6931)")
    for name, m in (("Up-leaning (p>=0.5)", p >= .5), ("Down-leaning (p<0.5)", p < .5)):
        print(f"  {name:<22} n={int(m.sum()):6d}  Brier {brier(p[m], y[m]):.4f}  mean predicted Up {p[m].mean():.3f}  realised Up {y[m].mean():.3f}")
    print("  by time into the window:")
    for a, b in ((0, 900), (900, 1800), (1800, 2700), (2700, 3451)):
        m = (prod[:, 1] >= a) & (prod[:, 1] < b)
        print(f"    {a // 60:2d}-{b // 60:2d} min: Brier {brier(p[m], y[m]):.4f}")
    print("  calibration (both directions):\n" + calib(p, y))

    # ---- 2. tune on train, judge on test
    print("\n== TUNING (score = Brier on TRAIN windows; then reported on held-out TEST windows) ==")
    grid = []
    for lag in (5, 10, 30, 60):
        for lb in (300, 900, 3600):
            for scale in (0.8, 0.9, 1.0, 1.1, 1.2, 1.35):
                for dist in ("normal", 6, 10):
                    grid.append((lag, lb, scale, dist))
    res = []
    for lag, lb, scale, dist in grid:
        sg = sigma_series(gf, lag, lb, 6.0, 300, scale)
        tr = evaluate(gf, s0, train, sg, dist)
        res.append((brier(tr[:, 3], tr[:, 4]), lag, lb, scale, dist))
    res.sort(key=lambda r: r[0])
    for b, lag, lb, scale, dist in res[:6]:
        sg = sigma_series(gf, lag, lb, 6.0, 300, scale)
        te = evaluate(gf, s0, test, sg, dist)
        print(f"  lag={lag:<3} lookback={lb:<5} scale={scale:<4} dist={str(dist):<6} train Brier {b:.4f}   TEST Brier {brier(te[:, 3], te[:, 4]):.4f}")
    base_sg = sigma_series(gf, 10, 900, 6.0, 300)
    bt = evaluate(gf, s0, test, base_sg)
    print(f"  [current production settings]                           TEST Brier {brier(bt[:, 3], bt[:, 4]):.4f}")
    b, lag, lb, scale, dist = res[0]
    best_sg = sigma_series(gf, lag, lb, 6.0, 300, scale)
    te = evaluate(gf, s0, test, best_sg, dist)
    print(f"\n  BEST on held-out test: Brier {brier(te[:, 3], te[:, 4]):.4f}, logloss {logloss(te[:, 3], te[:, 4]):.4f}")
    print("  held-out calibration:\n" + calib(te[:, 3], te[:, 4]))
    for name, m in (("Up-leaning", te[:, 3] >= .5), ("Down-leaning", te[:, 3] < .5)):
        print(f"  {name:<13} n={int(m.sum()):6d}  Brier {brier(te[m, 3], te[m, 4]):.4f}")
    json.dump({"lag": lag, "lookback": lb, "scale": scale, "dist": dist, "train_brier": b}, open(ROOT / "data" / "best_params.json", "w"))
    return s0, g, gf, wins, (lag, lb, scale, dist)


if __name__ == "__main__":
    main()
