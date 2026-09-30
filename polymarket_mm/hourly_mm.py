"""Market maker for the BTC 60-min Up/Down market on Polymarket US.  Paper by default; --live trades.

  python -m polymarket_mm.hourly_mm                 paper: real feeds, simulated fills, P&L per window
  python -m polymarket_mm.hourly_mm --preview-check validate order payloads against the real API (no orders placed)
  python -m polymarket_mm.hourly_mm --live          REAL orders
"""
import argparse
import logging
import math
import threading
import time
from dataclasses import dataclass
from decimal import Decimal

from dotenv import load_dotenv

from .brti import BrtiFeed, KalshiAuth
from .hourly import HourlyMarket, find_current
from .model import SecondSampler, fair_up
from .quoter import _ceil, _d, _floor
from .us_broker import (BUY_LONG, BUY_SHORT, SELL_LONG, SELL_SHORT, LiveBroker, PaperBroker, UsRest,
                        order_body)
from .us_feed import UsAuth, UsBookFeed

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HourlyConfig:
    size: float = 10.0            # contracts per quote
    max_pos: float = 50.0         # max net contracts either way
    min_half: float = 0.02        # minimum half-spread (probability units)
    skew: float = 0.03            # fair-value shift at max inventory
    react_seconds: float = 5.0    # widen by the fair-value move possible in this long
    pull_frac: float = 0.6        # cancel instantly if fair drifts this fraction of half-spread from what we quoted
    start_delay: float = 15.0     # no quoting until this long after the window opens (reference price settles)
    stop_before_end: float = 150.0  # stop quoting this long before expiry (gamma blows up)
    min_p: float = 0.06           # don't quote near-certain outcomes
    max_p: float = 0.94
    warmup_samples: int = 180     # seconds of BRTI history needed before trusting the vol estimate
    spike_sigmas: float = 3.0     # 10s BRTI move (in sigmas) that halts quoting
    halt_seconds: float = 10.0
    loop_hz: float = 4.0


@dataclass(frozen=True)
class DesiredQuote:
    intent: str
    price: float
    qty: float


def yes_quotes(*, fair: float, half: float, tick: float, pos: float, cfg: HourlyConfig,
               min_qty: float = 0.01) -> list[DesiredQuote]:
    """Quotes in YES-price space. Inventory is unwound first, new exposure only opened when flat that side:
       bid side: SELL_SHORT (close NO) if pos<0 else BUY_LONG;   ask side: SELL_LONG if pos>0 else BUY_SHORT."""
    t = _d(tick)
    skew = max(-1.0, min(1.0, pos / cfg.max_pos)) if cfg.max_pos > 0 else 0.0
    center = _d(fair) - _d(cfg.skew) * _d(skew)  # long inventory -> quote lower, want to sell
    bid = _floor(center - _d(half), t)
    ask = _ceil(center + _d(half), t)
    if bid < t or ask > 1 - t or bid >= ask:
        return []
    out: list[DesiredQuote] = []
    if pos < -min_qty:
        out.append(DesiredQuote(SELL_SHORT, float(bid), min(cfg.size, -pos)))
    elif cfg.max_pos - pos >= min_qty:
        out.append(DesiredQuote(BUY_LONG, float(bid), min(cfg.size, cfg.max_pos - pos)))
    if pos > min_qty:
        out.append(DesiredQuote(SELL_LONG, float(ask), min(cfg.size, pos)))
    elif cfg.max_pos + pos >= min_qty:
        out.append(DesiredQuote(BUY_SHORT, float(ask), min(cfg.size, cfg.max_pos + pos)))
    return out


class HourlyMaker:
    def __init__(self, cfg: HourlyConfig, us: UsAuth, brti: BrtiFeed, sampler: SecondSampler, live_rest: UsRest | None):
        self.cfg, self.us, self.brti, self.sampler, self.rest = cfg, us, brti, sampler, live_rest
        self.mkt: HourlyMarket | None = None
        self.book_feed: UsBookFeed | None = None
        self.broker = None
        self._lock = threading.Lock()  # serialises broker calls between the main loop and the BRTI thread
        self._quoted_fair: float | None = None
        self._quoted_half = 0.0
        self._halt_until = 0.0
        self.stop = threading.Event()
        self.total_pnl = 0.0
        brti.on_tick = self._on_tick

    # -- model inputs -------------------------------------------------------
    def _k(self) -> float | None:
        m = self.mkt
        return (m.price_to_beat or self.sampler.window_avg(m.window_start)) if m else None

    def _fair(self, now: float, spot: float, sigma: float) -> float:
        return fair_up(spot=spot, k=self._k(), now=now, window_end=self.mkt.window_end, sigma=sigma, sampler=self.sampler)

    def _half(self, now: float, spot: float, sigma: float, fair: float) -> float:
        """Half-spread = max(min_half, how far fair value could move within react_seconds)."""
        d = sigma * math.sqrt(self.cfg.react_seconds)
        hi = self._fair(now, spot + d, sigma)
        lo = self._fair(now, spot - d, sigma)
        return max(self.cfg.min_half, (hi - lo) / 2)

    def _quoteable(self, now: float) -> str | None:
        """Reason NOT to quote right now, else None."""
        m, c = self.mkt, self.cfg
        if not m.tradable:
            return "market not tradable"
        if now < m.window_start + c.start_delay:
            return "waiting for window open"
        if now > m.window_end - c.stop_before_end:
            return "final minutes"
        if self.brti.latest() is None:
            return "BRTI feed stale"
        if self.book_feed.book(m.slug) is None:
            return "no book"
        if len(self.sampler.px) < c.warmup_samples:
            return "vol warm-up"
        if self._k() is None:
            return "no reference price"
        if now < self._halt_until:
            return "spike halt"
        return None

    # -- event-driven: runs on the BRTI thread for every tick --------------------
    def _on_tick(self, t) -> None:
        if t.spot:
            self.sampler.add(t.local_ts, t.spot)
        if self.mkt is None or self.broker is None or self._quoted_fair is None or not t.spot:
            return
        now = time.time()
        try:
            sigma = self.sampler.sigma(now)
            fair = self._fair(now, t.spot, sigma)
            # 10s BRTI jump beyond spike_sigmas * sigma*sqrt(10)?
            prev = self.sampler.px.get(int(now) - 10)
            spiked = prev is not None and abs(t.spot - prev) > self.cfg.spike_sigmas * sigma * math.sqrt(10)
            stale = abs(fair - self._quoted_fair) > self.cfg.pull_frac * self._quoted_half
            if spiked or stale:
                with self._lock:
                    if self._quoted_fair is not None:
                        self.broker.cancel_all()
                        self._quoted_fair = None
                        if spiked:
                            self._halt_until = now + self.cfg.halt_seconds
                        log.info("INSTANT PULL (%s): fair %.3f vs quoted", "spike" if spiked else "fair moved", fair)
        except Exception:
            log.exception("tick handler failed")

    def _on_book(self, book) -> None:
        if isinstance(self.broker, PaperBroker):
            self.broker.on_book(book.best_bid, book.best_ask)

    # -- lifecycle ----------------------------------------------------------
    def _roll(self, now: float) -> bool:
        nxt = find_current(self.us, now)
        if nxt is None:
            return False
        if self.book_feed:
            self.book_feed.stop()
        self.mkt = nxt
        self.book_feed = UsBookFeed(self.us, [nxt.slug])
        self.book_feed.on_update = self._on_book
        self.book_feed.start()
        self.broker = LiveBroker(self.rest, nxt.slug) if self.rest else PaperBroker()
        self._quoted_fair = None
        log.info("market %s  (%s mode)", nxt.slug, "LIVE" if self.rest else "paper")
        return True

    def _settle_paper(self, now: float) -> None:
        m = self.mkt
        if not isinstance(self.broker, PaperBroker) or now < m.window_end + 3:
            return
        end_avg, k = self.sampler.window_avg(m.window_end), self._k()
        if end_avg is None or k is None:
            log.warning("cannot settle %s (missing BRTI data)", m.slug)
            return
        up = end_avg >= k
        fills = len(self.broker.fills)
        pnl = self.broker.settle(up)
        self.total_pnl += pnl
        log.info("SETTLED %s: end_avg=%.2f K=%.2f -> %s | fills=%d paper P&L=$%.2f | session total=$%.2f",
                 m.slug, end_avg, k, "UP" if up else "DOWN", fills, pnl, self.total_pnl)

    def _reconcile(self, want: list[DesiredQuote], tick: float) -> None:
        have = self.broker.open_orders()
        keep_ids, todo = set(), list(want)
        for o in have:
            hit = next((q for q in todo if q.intent == o.intent and abs(q.price - o.price) < tick / 2
                        and 0.5 * q.qty <= o.qty <= q.qty + 1e-9), None)
            if hit:
                todo.remove(hit)
                keep_ids.add(o.id)
        for o in have:
            if o.id not in keep_ids:
                self.broker.cancel(o.id)
        for q in todo:
            self.broker.place(q.intent, q.price, q.qty)

    def run(self) -> None:
        last_log = 0.0
        try:
            while not self.stop.is_set():
                now = time.time()
                if self.mkt is None or now >= self.mkt.window_end + 3:
                    if self.mkt:
                        with self._lock:
                            self.broker.cancel_all()
                        self._settle_paper(now)
                    if not self._roll(now):
                        time.sleep(5)
                        continue
                m = self.mkt
                if m.price_to_beat is None and now >= m.window_start + 10:
                    fresh = find_current(self.us, now)
                    if fresh and fresh.slug == m.slug and fresh.price_to_beat:
                        self.mkt = m = fresh
                why = self._quoteable(now)
                with self._lock:
                    if why:
                        if self._quoted_fair is not None or self.broker.open_orders():
                            self.broker.cancel_all()
                            self._quoted_fair = None
                        if now - last_log > 10:
                            log.info("not quoting: %s (T-%.0fs)", why, m.window_end - now)
                            last_log = now
                    else:
                        spot = self.sampler.last()
                        sigma = self.sampler.sigma(now)
                        fair = self._fair(now, spot, sigma)
                        if self.cfg.min_p <= fair <= self.cfg.max_p:
                            half = self._half(now, spot, sigma, fair)
                            want = yes_quotes(fair=fair, half=half, tick=m.tick, pos=self.broker.position(), cfg=self.cfg,
                                              min_qty=m.min_qty)
                            self._reconcile(want, m.tick)
                            self._quoted_fair, self._quoted_half = fair, half
                            if now - last_log > 5:
                                log.info("T-%4.0fs BRTI=%.2f K=%.2f fair=%.3f half=%.3f pos=%+g quotes=%s", m.window_end - now,
                                         spot, self._k(), fair, half, self.broker.position(),
                                         [(q.intent[13:], q.price) for q in want])
                                last_log = now
                        else:
                            self.broker.cancel_all()
                            self._quoted_fair = None
                time.sleep(1 / self.cfg.loop_hz)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("shutting down: cancelling all quotes")
        try:
            if self.broker:
                with self._lock:
                    self.broker.cancel_all()
        except Exception:
            log.exception("CANCEL FAILED on shutdown - CHECK OPEN ORDERS MANUALLY")
        if self.book_feed:
            self.book_feed.stop()
        self.brti.stop()


def preview_check(us: UsAuth) -> None:
    """Send order payloads to /v1/order/preview (documented side-effect free) to validate our format."""
    rest = UsRest(us)
    m = find_current(us)
    if not m:
        raise SystemExit("no live hourly market")
    print("market:", m.slug)
    for intent, price in ((BUY_LONG, 0.01), (BUY_SHORT, 0.99), (SELL_LONG, 0.99), (SELL_SHORT, 0.01)):
        body = order_body(m.slug, intent, price, 1)
        try:
            r = rest.preview(body)
            o = r.get("order", {})
            print(f"  {intent[13:]:<11} @ {price}: OK  state={o.get('state')} side={o.get('side')} price={o.get('price')}")
        except Exception as e:
            print(f"  {intent[13:]:<11} @ {price}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place REAL orders")
    ap.add_argument("--preview-check", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    for n in ("websockets", "urllib3"):
        logging.getLogger(n).setLevel(logging.WARNING)
    us = UsAuth.from_env()
    if args.preview_check:
        return preview_check(us)

    sampler = SecondSampler()
    brti = BrtiFeed(KalshiAuth.from_env())
    rest = UsRest(us) if args.live else None
    if args.live:
        log.warning("LIVE MODE: placing real orders")
    maker = HourlyMaker(HourlyConfig(), us, brti, sampler, rest)
    brti.start()
    try:
        maker.run()
    except KeyboardInterrupt:
        maker.stop.set()


if __name__ == "__main__":
    main()
