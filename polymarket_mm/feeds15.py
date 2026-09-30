"""Watch the BTC 15-minute Up/Down market on BOTH exchanges at once (read-only, places no orders).

  python -m polymarket_mm.feeds15

Both list the same contract: same 15-min window, same reference price (Kalshi floor_strike == Polymarket priceToBeat),
both settle on the average of 60 BRTI prints.  Shows BRTI, our fair value, each venue's Up bid/ask, and whether the two
venues are CROSSED (one's bid above the other's ask = a riskless gross gap, before fees).
"""
import argparse
import logging
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from .brti import BrtiFeed, KalshiAuth
from .hourly import find_current as find_pm
from .kalshi_feed import KalshiBookFeed
from .model import SecondSampler, fair_up
from .us_feed import UsAuth, UsBookFeed

log = logging.getLogger(__name__)


def fmt(b):
    return "  -- / --  " if not b or b.best_bid is None or b.best_ask is None else f"{b.best_bid:.3f} / {b.best_ask:.3f}"


def cross_gap(pm, ka):
    """Positive = crossed: buy Up on one venue and Down on the other for less than $1 (gross, before fees)."""
    if not pm or not ka or None in (pm.best_bid, pm.best_ask, ka.best_bid, ka.best_ask):
        return None
    return max(ka.best_bid - pm.best_ask, pm.best_bid - ka.best_ask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    for n in ("websockets", "urllib3"):
        logging.getLogger(n).setLevel(logging.WARNING)

    us, kal = UsAuth.from_env(), KalshiAuth.from_env()
    sampler = SecondSampler()
    from .hourly_mm import seed_sampler
    log.info("seeded %d s of BRTI history", seed_sampler(sampler, kal))
    brti = BrtiFeed(kal)
    brti.on_tick = lambda t: t.spot and sampler.add(t.local_ts, t.spot)
    brti.start()
    kfeed = KalshiBookFeed(kal)
    kfeed.start()

    pm_mkt, pm_feed = None, None
    try:
        while True:
            now = time.time()
            if pm_mkt is None or now >= pm_mkt.window_end:
                nxt = find_pm(us, now, horizon="15m")
                if nxt is None:
                    time.sleep(3)
                    continue
                if pm_feed:
                    pm_feed.stop()
                pm_mkt, pm_feed = nxt, UsBookFeed(us, [nxt.slug])
                pm_feed.start()
                log.info("Polymarket 15m market: %s  window %s-%s UTC  ref %s", nxt.slug,
                         datetime.fromtimestamp(nxt.window_start, timezone.utc).strftime("%H:%M"),
                         datetime.fromtimestamp(nxt.window_end, timezone.utc).strftime("%H:%M"), nxt.price_to_beat)
            km, pb, kb = kfeed.market, pm_feed.book(pm_mkt.slug), kfeed.book()
            spot = sampler.last()
            ref = pm_mkt.price_to_beat or (km.reference if km else None)
            fair = None
            if spot and ref:
                fair = fair_up(spot=spot, k=ref, now=now, window_end=pm_mkt.window_end, sigma=sampler.sigma(now), sampler=sampler)
            g = cross_gap(pb, kb)
            same = "" if not km else ("" if abs(km.reference - (pm_mkt.price_to_beat or km.reference)) < 0.005 else "  !REF MISMATCH")
            print(f"T-{pm_mkt.window_end - now:4.0f}s  BRTI {spot or 0:>9.2f}  ref {ref or 0:>9.2f}  fair UP {('%.3f' % fair) if fair is not None else ' -- '}"
                  f"  |  POLYMARKET {fmt(pb)}  |  KALSHI {fmt(kb)}"
                  f"  |  cross {('%+.3f' % g) if g is not None else ' -- '}{'  <-- CROSSED' if g and g > 0 else ''}{same}", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        kfeed.stop(); brti.stop()
        if pm_feed:
            pm_feed.stop()


if __name__ == "__main__":
    main()
