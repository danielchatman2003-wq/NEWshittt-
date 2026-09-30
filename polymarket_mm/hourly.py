"""BTC 60-minute Up/Down on Polymarket US: discovery + live model-vs-book monitor (dry-run).

  python -m polymarket_mm.hourly            watch the current market, roll to the next automatically
"""
import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from .brti import BrtiFeed, KalshiAuth
from .model import SecondSampler, fair_up
from .us_feed import REST, UsAuth, UsBookFeed

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HourlyMarket:
    slug: str
    window_start: float  # unix seconds
    window_end: float
    price_to_beat: float | None  # None until the exchange stamps it (a few s after the open)
    tick: float
    min_qty: float
    fee_coef: float
    tradable: bool


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def parse_updown(m: dict, horizon: str = "1h") -> HourlyMarket | None:
    """Accept only a BTC / BRTI / UP_DOWN market of the given horizon ("1h" or "15m"), validated by fields (not the slug)."""
    t = m.get("assetPriceTerms") or {}
    if (t.get("marketType") != "ASSET_PRICE_MARKET_TYPE_UP_DOWN" or t.get("indexSymbol") != "BRTI"
            or t.get("horizon") != horizon or (t.get("asset") or {}).get("symbol") != "btc"):
        return None
    ptb = (t.get("priceToBeat") or {}).get("value")
    return HourlyMarket(
        slug=m["slug"], window_start=_ts(t["windowStart"]), window_end=_ts(t["windowEnd"]),
        price_to_beat=float(ptb) if ptb else None, tick=float(m.get("orderPriceMinTickSize") or 0.01),
        min_qty=float(m.get("minimumTradeQty") or 0.01), fee_coef=float(m.get("feeCoefficient") or 0.0695),
        tradable=bool(m.get("active")) and not m.get("closed") and m.get("status") == "MARKET_STATUS_OPEN",
    )


def parse_hourly(m: dict) -> HourlyMarket | None:
    return parse_updown(m, "1h")


def candidate_slugs(now: float, horizon: str = "1h") -> list[str]:
    """Slugs of the window containing `now` and the next one. (Docs call the slug format non-contractual, so every
    candidate is validated by its fields after fetching.)"""
    step = {"1h": 60, "15m": 15}[horizon]
    t = datetime.fromtimestamp(now, timezone.utc)
    base = t.replace(minute=(t.minute // step) * step if step < 60 else 0, second=0, microsecond=0)
    out = []
    for k in (0, 1):
        h = base + timedelta(minutes=step * k)
        out.append(f"cpc-btc-updown-{horizon}-{h:%Y-%m-%d-%H%M}z" if horizon == "15m" else f"cpc-btc-updown-1h-{h:%Y-%m-%d-%H}00z")
    return out


def find_current(auth: UsAuth, now: float | None = None, session=None, horizon: str = "1h") -> HourlyMarket | None:
    """The market (1h or 15m) whose window contains `now` (else the next one that hasn't started)."""
    s = session or requests
    now = now or time.time()
    for slug in candidate_slugs(now, horizon):
        path = f"/v1/market/slug/{slug}"
        r = s.get(REST + path, headers=auth.headers("GET", path), timeout=10)
        if r.status_code == 200:
            hm = parse_updown(r.json().get("market", {}), horizon)
            if hm and hm.window_end > now:
                return hm
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    for n in ("websockets", "urllib3"):
        logging.getLogger(n).setLevel(logging.WARNING)

    us = UsAuth.from_env()
    sampler = SecondSampler()
    brti = BrtiFeed(KalshiAuth.from_env())
    brti.on_tick = lambda t: t.spot and sampler.add(t.local_ts, t.spot)
    brti.start()

    mkt, book_feed = None, None
    try:
        while True:
            now = time.time()
            if mkt is None or now >= mkt.window_end:
                nxt = find_current(us, now)
                if nxt is None:
                    log.info("no live hourly market found, retrying")
                    time.sleep(5)
                    continue
                if book_feed:
                    book_feed.stop()
                mkt = nxt
                book_feed = UsBookFeed(us, [mkt.slug])
                book_feed.start()
                log.info("market %s  window %s -> %s UTC  fee=%.4f", mkt.slug,
                         datetime.fromtimestamp(mkt.window_start, timezone.utc).strftime("%H:%M"),
                         datetime.fromtimestamp(mkt.window_end, timezone.utc).strftime("%H:%M"), mkt.fee_coef)

            if mkt.price_to_beat is None and now >= mkt.window_start + 10:  # refresh once the exchange stamps it
                fresh = find_current(us, now)
                if fresh and fresh.slug == mkt.slug and fresh.price_to_beat:
                    mkt = fresh
            spot = sampler.last()
            k = mkt.price_to_beat or sampler.window_avg(mkt.window_start)
            book = book_feed.book(mkt.slug)
            if spot and k:
                sig = sampler.sigma(now)
                p = fair_up(spot=spot, k=k, now=now, window_end=mkt.window_end, sigma=sig, sampler=sampler)
                bid, ask = (book.best_bid, book.best_ask) if book else (None, None)
                mid = book.mid if book else None
                edge = f"{(p - mid) * 100:+.1f}c" if mid is not None else "n/a"
                print(f"T-{mkt.window_end - now:5.0f}s  BRTI={spot:.2f}  K={k:.2f} ({spot - k:+.2f})  sigma={sig:.2f}$/√s  "
                      f"FAIR_UP={p:.3f}  book={bid}/{ask}  model-mid={edge}", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        brti.stop()
        if book_feed:
            book_feed.stop()


if __name__ == "__main__":
    main()
