"""Market maker for the BTC 60-min Up/Down market on Polymarket US.  Paper by default; --live trades.

  python -m polymarket_mm.hourly_mm                 paper: real feeds, simulated fills, P&L per window
  python -m polymarket_mm.hourly_mm --preview-check validate order payloads against the real API (no orders placed)
  python -m polymarket_mm.hourly_mm --live          REAL orders
"""
import argparse
from collections import deque
import logging
import math
import signal
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
    size: float = 1.0             # contracts per quote (1 per trade)
    max_pos: float = 1.0          # max net contracts either way: at 1, a held contract only quotes its EXIT (no stacking)
    min_half: float = 0.02        # minimum half-spread (probability units)
    skew: float = 0.03            # fair-value shift at max inventory
    react_seconds: float = 5.0    # widen by the fair-value move possible in this long
    pull_frac: float = 0.6        # cancel instantly if fair drifts this fraction of half-spread from what we quoted
    start_delay: float = 900.0    # no quoting for the first 15 min: backtest shows the model is no better than the market there
                                  # (Brier 0.2368 vs 0.2350) and clearly better later, so early quotes just get picked off
    stop_before_end: float = 150.0  # stop quoting this long before expiry (gamma blows up)
    min_p: float = 0.06           # don't quote near-certain outcomes
    max_p: float = 0.94
    warmup_samples: int = 180     # seconds of BRTI history needed before trusting the vol estimate
    spike_sigmas: float = 3.0     # 10s BRTI move (in sigmas) that halts quoting
    halt_seconds: float = 10.0
    fill_widen: float = 0.01      # widen both sides this much after any fill (fills are where informed flow shows up)
    fill_widen_seconds: float = 30.0
    trend_window: float = 3.0     # also widen by how far fair value moved over this many seconds
    trend_cap: float = 0.05
    exit_slack: float = 0.02      # an exit may sell up to this far below fair (to join the book and actually fill)
    touch_tol: float = 0.02       # join the book's top on both sides if within this of fair (0 = model-edge quotes only)
    exit_hold: float = 30.0       # an exit order rests at least this long before it may be moved DOWN/passive (it may still move
                                  # up immediately if it would be selling too cheap). Chasing a falling market never fills.
    exit_take_after: float = 60.0  # still holding after this long -> cross the spread (IOC) to get out
    take_slack: float = 0.05      # ...but never take an exit more than this below fair value
    take_retry: float = 10.0
    exit_ticks: int = 3           # exits are sticky: keep a resting exit up to this many ticks more passive than desired
    loop_hz: float = 4.0
    max_errors: int = 40          # consecutive failed iterations (~1 min of backoff) before giving up


@dataclass(frozen=True)
class DesiredQuote:
    intent: str
    price: float
    qty: float


def yes_quotes(*, fair: float, half: float, tick: float, pos: float, cfg: HourlyConfig,
               min_qty: float = 0.01, book: tuple[float | None, float | None] | None = None,
               touch_tol: float = 0.0) -> list[DesiredQuote]:
    """Quotes in YES-price space. Inventory is unwound first, new exposure only opened when flat that side:
       bid side: SELL_SHORT (close NO) if pos<0 else BUY_LONG;   ask side: SELL_LONG if pos>0 else BUY_SHORT."""
    t = _d(tick)
    skew = max(-1.0, min(1.0, pos / cfg.max_pos)) if cfg.max_pos > 0 else 0.0
    # round away float dust (e.g. 0.50000000004) so a dead-even market quotes dead-even, not tilted a tick toward Up
    center = _d(round(fair, 6)) - _d(cfg.skew) * _d(skew)  # long inventory -> quote lower, want to sell
    bid = _floor(center - _d(half), t)
    ask = _ceil(center + _d(half), t)
    if bid < t or ask > 1 - t or bid >= ask:
        return []
    # Two-sided flow: ENTRY quotes join the top of the real book on BOTH sides (bid Up at the best bid, bid Down at the
    # best Down bid = Up ask), as long as that price is within `touch_tol` of fair. Without this, when the book is a bit
    # more bearish than the model the Up bid lands at the top of the book and fills, while the Down bid sits cents behind
    # the Down book and never does -- a one-sided machine. The model still caps what we will pay (fair + touch_tol).
    if book and touch_tol > 0 and book[0] is not None and book[1] is not None:
        cap_bid = _floor(_d(round(fair, 6)) + _d(touch_tol), t)       # dearest we'll pay for Up (or sell Down)
        cap_ask = _ceil(_d(round(fair, 6)) - _d(touch_tol), t)        # cheapest we'll sell Up / pay for Down
        tb = max(bid, min(_floor(_d(book[0]), t), cap_bid))           # join best bid if the model allows
        ta = min(ask, max(_ceil(_d(book[1]), t), cap_ask))            # join best ask if the model allows
        tb = min(tb, _d(book[1]) - t)                                  # post-only: never at/through the other side
        ta = max(ta, _d(book[0]) + t)
        if tb < ta:
            bid, ask = tb, ta
    # Exits are priced off the REAL book (best_bid, best_ask of the YES book): join the top of the book on our side so
    # we actually get filled, but never give away more than exit_slack below fair, and never cross (post-only).
    bb, ba = (book or (None, None))
    if bb is not None and ba is not None:
        slack = _d(cfg.exit_slack)
        lo_ask = _ceil(_d(round(fair, 6)) - slack, t)                 # lowest price we'll sell Up for
        exit_ask = min(ask, max(_ceil(_d(ba), t), lo_ask))            # join best ask, bounded by slack and our normal ask
        exit_ask = max(exit_ask, _d(bb) + t)                          # never at/below the best bid (would cross)
        hi_bid = _floor(_d(round(fair, 6)) + slack, t)                # highest price we'll pay to buy back Up / sell Down
        exit_bid = max(bid, min(_floor(_d(bb), t), hi_bid))           # join best bid, bounded by slack and our normal bid
        exit_bid = min(exit_bid, _d(ba) - t)
        exit_bid, exit_ask = (exit_bid if exit_bid < exit_ask else bid), (exit_ask if exit_bid < exit_ask else ask)
    else:
        exit_bid, exit_ask = bid, ask
    out: list[DesiredQuote] = []
    if pos < -min_qty:
        out.append(DesiredQuote(SELL_SHORT, float(exit_bid), min(cfg.size, -pos)))
    elif cfg.max_pos - pos >= min_qty:
        out.append(DesiredQuote(BUY_LONG, float(bid), min(cfg.size, cfg.max_pos - pos)))
    if pos > min_qty:
        out.append(DesiredQuote(SELL_LONG, float(exit_ask), min(cfg.size, pos)))
    elif cfg.max_pos + pos >= min_qty:
        out.append(DesiredQuote(BUY_SHORT, float(ask), min(cfg.size, cfg.max_pos + pos)))
    return out


def exit_take(*, pos: float, fair: float, best_bid: float, best_ask: float, tick: float, cfg: HourlyConfig,
              min_qty: float = 0.01):
    """Cross-the-spread exit for stale inventory: returns (intent, worst_yes_price, qty) or None.
    Up held:   SELL_LONG  IOC with limit best_bid - 1 tick.   Down held: SELL_SHORT IOC with limit best_ask + 1 tick
    (prices are YES prices). Refused if it would give up more than take_slack versus fair value."""
    if abs(pos) <= min_qty:
        return None
    t = _d(tick)
    if pos > 0:
        limit = _floor(_d(best_bid) - t, t)
        if limit < t or float(limit) < fair - cfg.take_slack:
            return None
        return SELL_LONG, float(limit), min(cfg.size, pos)
    limit = _ceil(_d(best_ask) + t, t)
    if limit > 1 - t or float(limit) > fair + cfg.take_slack:
        return None
    return SELL_SHORT, float(limit), min(cfg.size, -pos)


def extra_half(*, now: float, last_fill: float, fair_hist, fair: float, cfg: HourlyConfig) -> float:
    """Adverse-selection padding on top of the base half-spread: (a) +fill_widen for fill_widen_seconds after any
    fill, (b) the amount fair value moved over the last trend_window seconds (capped). Fast markets and fresh
    fills are exactly when informed traders are hitting us, so we stand further away."""
    pad = cfg.fill_widen if now - last_fill < cfg.fill_widen_seconds else 0.0
    old = next((f for t, f in fair_hist if t >= now - cfg.trend_window), None)
    if old is not None:
        pad += min(cfg.trend_cap, abs(fair - old))
    return pad


def describe(q: DesiredQuote) -> str:
    """Plain-English quote: prices are shown in the price OF THE SIDE being traded (Up or Down)."""
    return {BUY_LONG: f"BID UP {q.price:.2f}", SELL_LONG: f"SELL UP {q.price:.2f}",
            BUY_SHORT: f"BID DOWN {1 - q.price:.2f}", SELL_SHORT: f"SELL DOWN {1 - q.price:.2f}"}[q.intent]


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
        self._fair_hist: deque = deque(maxlen=64)   # (time, fair) for the trend pad
        self._last_fill = 0.0
        self._prev_pos: float | None = None
        self._last_log = 0.0
        self._inv_since: float | None = None   # when the current inventory was first seen
        self._last_take = 0.0
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
        if self.rest:
            self.broker = LiveBroker(self.rest, nxt.slug, expire_at=nxt.window_end - self.cfg.stop_before_end)
            self.broker.cancel_all()  # start clean: nothing of ours resting from a previous run
            log.warning("LIVE on %s: position %+g, buying power $%s", nxt.slug, self.broker.position(), self.broker._bp)
        else:
            self.broker = PaperBroker()
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

    def _reconcile(self, want: list[DesiredQuote], tick: float, fair: float, half: float) -> None:
        """Make resting orders match `want` without churning. A resting order is kept while it sits where we would
        put it now, allowing it to be up to ONE tick more passive than desired (a 1-cent wiggle doesn't burn the
        rate limit or our queue spot) but never MORE aggressive than desired (it would be giving edge away).
        Judged against the desired price only: exit quotes are deliberately priced at/below fair to shed inventory,
        so an 'edge vs fair' test would wrongly replace them every cycle."""
        have = self.broker.open_orders()
        keep_ids, todo = set(), list(want)
        eps = tick * 0.01
        cfg = getattr(self, 'cfg', None) or HourlyConfig()
        for o in have:
            for q in todo:
                if q.intent != o.intent or not (0.5 * q.qty <= o.qty <= q.qty + 1e-9):
                    continue
                bidlike = o.intent in (BUY_LONG, SELL_SHORT)          # rests below the market in YES-price space
                passive_by = (q.price - o.price) if bidlike else (o.price - q.price)   # >0: resting is more passive
                is_exit = o.intent in (SELL_LONG, SELL_SHORT)
                room = (cfg.exit_ticks if is_exit else 1) * tick   # exits hold their place in the queue
                age = (time.monotonic() - o.placed_at) if o.placed_at else 1e9
                young_exit = is_exit and age < cfg.exit_hold       # a young exit is NEVER moved passive/down (chasing a
                                                                   # falling market means it never rests long enough to fill)
                if passive_by >= -eps and (passive_by <= room + eps or young_exit):
                    todo.remove(q)
                    keep_ids.add(o.id)
                    break
        for o in have:
            if o.id not in keep_ids:
                self.broker.cancel(o.id)
        for q in todo:
            self.broker.place(q.intent, q.price, q.qty)

    def _step(self) -> None:
        """One quoting iteration. May raise on network errors; run() handles that."""
        now = time.time()
        if self.mkt is None or now >= self.mkt.window_end + 3:
            if self.mkt:
                with self._lock:
                    self.broker.cancel_all()
                self._settle_paper(now)
            if not self._roll(now):
                time.sleep(5)
                return
        m = self.mkt
        if m.price_to_beat is None and now >= m.window_start + 10:
            fresh = find_current(self.us, now)
            if fresh and fresh.slug == m.slug and fresh.price_to_beat:
                self.mkt = m = fresh
        fault = getattr(self.broker, "fault", None)
        if fault:
            log.critical("BROKER FAULT: %s", fault)
            self.stop.set()
            return
        why = self._quoteable(now)
        with self._lock:
            if why:
                if self._quoted_fair is not None or self.broker.open_orders():
                    self.broker.cancel_all()
                    self._quoted_fair = None
                if now - self._last_log > 10:
                    log.info("not quoting: %s (T-%.0fs)", why, m.window_end - now)
                    self._last_log = now
                return
            spot = self.sampler.last()
            sigma = self.sampler.sigma(now)
            fair = self._fair(now, spot, sigma)
            if not (self.cfg.min_p <= fair <= self.cfg.max_p):
                self.broker.cancel_all()
                self._quoted_fair = None
                return
            pos = self.broker.position()
            if self._prev_pos is not None and abs(pos - self._prev_pos) > 1e-9:
                self._last_fill = now  # position changed => we were just traded against
            self._prev_pos = pos
            if abs(pos) > m.min_qty:
                self._inv_since = self._inv_since or now
            else:
                self._inv_since = None
            bk = self.book_feed.book(m.slug)
            if (bk and self._inv_since and now - self._inv_since >= self.cfg.exit_take_after
                    and now - self._last_take >= self.cfg.take_retry and bk.best_bid is not None and bk.best_ask is not None):
                tk = exit_take(pos=pos, fair=fair, best_bid=bk.best_bid, best_ask=bk.best_ask, tick=m.tick, cfg=self.cfg,
                               min_qty=m.min_qty)
                self._last_take = now
                if tk:
                    log.warning("EXIT TAKE: held %+g for %.0fs without selling -> crossing the spread (%s limit %.2f)",
                                pos, now - self._inv_since, tk[0][13:], tk[1])
                    self.broker.cancel_all()          # free the position locked by our resting exit first
                    self._quoted_fair = None
                    self.broker.take(*tk)
                    return
                log.info("exit take skipped: bid too far below fair (model says hold)")
            pad = extra_half(now=now, last_fill=self._last_fill, fair_hist=self._fair_hist, fair=fair, cfg=self.cfg)
            half = self._half(now, spot, sigma, fair) + pad
            self._fair_hist.append((now, fair))
            want = yes_quotes(fair=fair, half=half, tick=m.tick, pos=pos, cfg=self.cfg, min_qty=m.min_qty,
                              book=(bk.best_bid, bk.best_ask) if bk else None,
                              touch_tol=max(0.0, self.cfg.touch_tol - pad))
            self._reconcile(want, m.tick, fair, half)
            self._quoted_fair, self._quoted_half = fair, half
            if now - self._last_log > 5:
                log.info("T-%4.0fs BRTI=%.2f ref=%.2f | fair UP %.3f / DOWN %.3f | pos %+g (+=Up -=Down) | %s",
                         m.window_end - now, spot, self._k(), fair, 1 - fair, pos,
                         "  &  ".join(describe(q) for q in want))
                self._last_log = now

    def run(self) -> None:
        """Quote until stopped. A network blip must never kill the bot: errors are logged and retried; if they
        persist we try hard to cancel everything and exit (orders also carry an exchange-side expiry)."""
        errors = 0
        try:
            while not self.stop.is_set():
                try:
                    self._step()
                    errors = 0
                except Exception as e:
                    errors += 1
                    log.warning("loop error %d/%d: %s: %s", errors, self.cfg.max_errors, type(e).__name__, str(e)[:120])
                    if errors == 3:  # stop leaving quotes unattended while the API is unreachable
                        self._safe_cancel_all("3 consecutive errors")
                    if errors >= self.cfg.max_errors:
                        log.critical("%d consecutive errors, giving up", errors)
                        break
                    time.sleep(min(2.0, 0.25 * errors))
                    continue
                time.sleep(1 / self.cfg.loop_hz)
        finally:
            self.shutdown()

    def _safe_cancel_all(self, why: str, attempts: int = 5) -> bool:
        if not self.broker:
            return True
        for i in range(attempts):
            try:
                with self._lock:
                    self.broker.cancel_all()
                self._quoted_fair = None
                return True
            except Exception as e:
                log.warning("cancel_all failed (%s) attempt %d/%d: %s", why, i + 1, attempts, str(e)[:100])
                time.sleep(1.0)
        return False

    def shutdown(self) -> None:
        log.info("shutting down: cancelling all quotes")
        if not self._safe_cancel_all("shutdown", attempts=8):
            log.critical("COULD NOT CANCEL ON SHUTDOWN - CHECK OPEN ORDERS MANUALLY (they expire at the quoting deadline)")
        if self.book_feed:
            self.book_feed.stop()
        self.brti.stop()


def seed_sampler(sampler: SecondSampler, kalshi: KalshiAuth, minutes: int = 20) -> int:
    """Pre-load the last ~`minutes` of real BRTI (one print per second) from Kalshi's CF history passthrough."""
    import requests
    from datetime import datetime, timedelta, timezone
    path, now = "/trade-api/v2/cfbenchmarks/history/values", time.time()
    base = datetime.fromtimestamp(now, timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours = [base - timedelta(hours=1), base] if (now - base.timestamp()) < minutes * 60 else [base]
    n = 0
    for h in hours:
        r = requests.get("https://external-api.kalshi.com" + path,
                         params={"id": "BRTI", "timespan": "HOUR", "timestamp": h.strftime("%Y-%m-%dT%H:%M:%SZ")},
                         headers=kalshi.headers("GET", path), timeout=30)
        if not r.ok:
            log.warning("could not seed BRTI history (%s); will warm up live instead", r.status_code)
            continue
        for t in r.json()["data"]["payload"]:
            if t["time"] % 1000 == 0 and t["time"] / 1000 <= now and t["time"] / 1000 >= now - minutes * 60:
                sampler.add(t["time"] / 1000, float(t["value"]))
                n += 1
    return n


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
    kalshi = KalshiAuth.from_env()
    log.info("seeded %d seconds of recent BRTI history (no warm-up wait)", seed_sampler(sampler, kalshi))
    brti = BrtiFeed(kalshi)
    rest = UsRest(us) if args.live else None
    if args.live:
        log.warning("LIVE MODE: placing real orders")
    maker = HourlyMaker(HourlyConfig(), us, brti, sampler, rest)
    brti.start()
    # SIGTERM (kill, docker stop, timeout) must cancel resting orders just like Ctrl-C does
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: maker.stop.set())
    maker.run()  # its finally-block cancels all quotes on exit


if __name__ == "__main__":
    main()
