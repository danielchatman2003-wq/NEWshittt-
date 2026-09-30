"""Validate against Polymarket US itself.

 (1) Do OUR window averages (from Kalshi's BRTI) reproduce the exchange's priceToBeat / settlementPrice / outcome?
 (2) Is the model better than the MARKET's own price at the same moments?   (the real test of "is fair value good")

  python research/validate.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import backtest as bt  # noqa: E402
from polymarket_mm.us_broker import UsRest  # noqa: E402
from polymarket_mm.us_feed import UsAuth  # noqa: E402

CACHE = ROOT / "data" / "market"
GATEWAY = "https://gateway.polymarket.us"


def slug_for(t0: int) -> str:
    return f"cpc-btc-updown-1h-{datetime.fromtimestamp(t0, timezone.utc):%Y-%m-%d-%H}00z"


def get_cached(name, fn):
    f = CACHE / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    v = fn()
    f.write_text(json.dumps(v))
    return v


def market_series(history, offs, t0):
    """Up-mid carried forward to each eval time: (long_ask + (1 - short_ask)) / 2, NaN before first quote."""
    ts = np.array([h["timestamp"] for h in history])
    mid = np.array([(h["longPrice"] + 1 - h["shortPrice"]) / 2 for h in history])
    out = np.full(len(offs), np.nan)
    for i, o in enumerate(offs):
        j = np.searchsorted(ts, t0 + o, side="right") - 1
        if j >= 0:
            out[i] = mid[j]
    return out


def main():
    load_dotenv()
    CACHE.mkdir(parents=True, exist_ok=True)
    rest = UsRest(UsAuth.from_env(), rps=8)
    s0, g = bt.load()
    gf = bt.ffill(g)
    wins = bt.windows(g, s0)
    dec = json.loads((ROOT / "data" / "best_params.json").read_text()) if (ROOT / "data" / "best_params.json").exists() else None

    # ---- (1) settlement reproduction
    k_err, f_err, agree, n = [], [], 0, 0
    usable = []
    for w in wins:
        slug = slug_for(w["t0"])
        try:
            m = get_cached("m_" + slug, lambda: rest.call("GET", f"/v1/market/slug/{slug}")["market"])
        except Exception:
            continue
        t = m.get("assetPriceTerms") or {}
        ptb, sp = (t.get("priceToBeat") or {}).get("value"), (t.get("settlementPrice") or {}).get("value")
        if not ptb or not sp or m.get("status") != "MARKET_STATUS_RESOLVED":
            continue
        n += 1
        k_err.append(w["k"] - float(ptb))
        f_err.append(w["f"] - float(sp))
        ex_up = json.loads(m["outcomePrices"])[0] in ("1", "1.0", "1.00")
        agree += int(ex_up == bool(w["up"]))
        usable.append((w, slug, float(ptb)))
    print("== (1) Does Kalshi BRTI reproduce Polymarket US's own numbers? ==")
    if n:
        ke, fe = np.abs(k_err), np.abs(f_err)
        print(f"  windows compared: {n}")
        print(f"  |our reference price - priceToBeat|: median ${np.median(ke):.3f}  p95 ${np.percentile(ke, 95):.3f}  max ${ke.max():.3f}")
        print(f"  |our end average  - settlementPrice|: median ${np.median(fe):.3f}  p95 ${np.percentile(fe, 95):.3f}  max ${fe.max():.3f}")
        print(f"  outcome (Up/Down) agreement: {agree}/{n}\n")

    # ---- (2) model vs market
    print("== (2) Model vs the MARKET's own price, same moments (walk-forward, no lookahead) ==")
    sig = bt.sigma_series(gf, 10, 900, 6.0, 300)
    offs = np.arange(15, bt.HOUR - 150 + 1, 30)
    P, M, Y, T, W = [], [], [], [], []
    for w, slug, ptb in usable:
        try:
            hist = get_cached("h_" + slug, lambda: _hist(rest, slug, w["t0"]))
        except Exception:
            continue
        if not hist:
            continue
        idx = w["t0"] - s0 + offs
        sd = sig[idx] * np.sqrt(np.maximum(bt.HOUR - offs - 40, 1))
        p = stats.norm.cdf((gf[idx] - ptb) / sd)       # use the EXCHANGE's reference price for a clean comparison
        mk = market_series(hist, offs, w["t0"])
        ok = ~np.isnan(mk)
        P.append(p[ok]); M.append(mk[ok]); Y.append(np.full(ok.sum(), w["up"])); T.append(offs[ok]); W.append(np.full(ok.sum(), len(W)))
    if not P:
        print("  no overlapping market history")
        return
    P, M, Y, T, W = map(np.concatenate, (P, M, Y, T, W))
    nw = len(usable)
    print(f"  {len(P)} comparison points from {nw} windows")
    print(f"  Brier  model {bt.brier(P, Y):.4f}   market {bt.brier(M, Y):.4f}   coin-flip 0.2500")
    print(f"  logloss model {bt.logloss(P, Y):.4f}   market {bt.logloss(M, Y):.4f}")
    for a, b in ((0, 900), (900, 1800), (1800, 2700), (2700, 3451)):
        m = (T >= a) & (T < b)
        print(f"    {a // 60:2d}-{b // 60:2d} min:  model {bt.brier(P[m], Y[m]):.4f}   market {bt.brier(M[m], Y[m]):.4f}")
    # resample whole WINDOWS (points inside a window are highly correlated) to see if the gap is real
    rng, nwin = np.random.default_rng(0), int(W.max()) + 1
    def boot(mask, label):
        d = (P - Y) ** 2 - (M - Y) ** 2                     # <0 => model better
        per = np.array([d[(W == i) & mask].mean() if ((W == i) & mask).any() else np.nan for i in range(nwin)])
        per = per[~np.isnan(per)]
        means = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(3000)]
        lo, hi = np.percentile(means, [2.5, 97.5])
        print(f"  Brier(model) - Brier(market), {label}: {per.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   -> "
              + ("model better (significant)" if hi < 0 else "market better (significant)" if lo > 0 else "NOT significantly different"))
    print("  blend w*model + (1-w)*market  (Brier, lower is better):")
    for wgt in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"    w={wgt:.2f}  {bt.brier(wgt * P + (1 - wgt) * M, Y):.4f}")
    boot(np.ones(len(P), bool), "whole window   ")
    boot(T < 1800, "first 30 min   ")
    boot(T >= 1800, "last 30 min    ")
    print("  who is right when they DISAGREE?  (edge = model - market; realised = outcome - market)")
    d = P - M
    for lo, hi in ((-1, -.10), (-.10, -.05), (-.05, -.02), (-.02, .02), (.02, .05), (.05, .10), (.10, 1)):
        m = (d >= lo) & (d < hi)
        if m.sum() > 20:
            print(f"    model-market in [{lo:+.2f},{hi:+.2f})  n={int(m.sum()):5d}  mean edge {d[m].mean():+.3f}   realised outcome - market {Y[m].mean() - M[m].mean():+.3f}")


def _hist(rest, slug, t0):
    path = "/v1/price-history"
    r = rest.s.get(GATEWAY + path, params={"symbol": slug, "fidelity": 1, "timestamp.startTimestamp": t0,
                                           "timestamp.endTimestamp": t0 + bt.HOUR},
                   headers=rest.auth.headers("GET", path), timeout=20)
    rest.limiter.acquire()
    return r.json().get("history", []) if r.ok else []


if __name__ == "__main__":
    main()
