"""Does recent TREND (minutes scale) predict the outcome beyond what the model's fair value already says?

For each checkpoint: residual = outcome - model_p. Regress it on the past k-minute BRTI change in units of
expected noise (change / (sigma*sqrt(k*60))).  slope>0 => trends persist (model too slow), ~0 => model already
has it.  Confidence from a bootstrap that resamples whole WINDOWS.      python research/trend.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402


def main():
    s0, g = bt.load()
    gf = bt.ffill(g)
    wins = bt.windows(g, s0)
    sig = bt.sigma_series(gf, 10, 900, 6.0, 300)
    offs = np.arange(900, bt.HOUR - 150 + 1, 30)          # the bot only quotes from minute 15
    rng = np.random.default_rng(3)
    print(f"{len(wins)} windows, checkpoints every 30s from minute 15 to 57\n")
    print(f"{'lookback':>9} {'slope of (outcome - p) on trend z':>36}   95% CI (bootstrap over windows)   verdict")
    for k in (60, 180, 300, 600, 900):
        X, R, W = [], [], []
        for wi, w in enumerate(wins):
            idx = w["t0"] - s0 + offs
            sd = sig[idx] * np.sqrt(np.maximum(bt.HOUR - offs - 40, 1))
            p = bt.stats.norm.cdf((gf[idx] - w["k"]) / sd)
            z = (gf[idx] - gf[idx - k]) / (sig[idx] * np.sqrt(k))
            X.append(z); R.append(w["up"] - p); W.append(np.full(len(idx), wi))
        X, R, W = map(np.concatenate, (X, R, W))
        def slope(mask_idx):
            x, r = X[mask_idx], R[mask_idx]
            return np.cov(x, r)[0, 1] / x.var()
        full = slope(np.arange(len(X)))
        per = [np.where(W == i)[0] for i in range(len(wins))]
        boots = []
        for _ in range(1500):
            pick = rng.integers(0, len(wins), len(wins))
            boots.append(slope(np.concatenate([per[i] for i in pick])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "trend persists" if lo > 0 else "trend REVERSES" if hi < 0 else "no signal beyond the model"
        print(f"{k // 60:>6} min {full:>+36.4f}   [{lo:+.4f}, {hi:+.4f}]   {verdict}")
    print("\n(slope units: change in P(Up) per 1-sigma of recent trend; 0.05 would mean a 1-sigma down-trend lowers P(Up) by 5c beyond the model)")


if __name__ == "__main__":
    main()
