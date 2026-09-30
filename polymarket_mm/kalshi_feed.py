"""Kalshi order-book websocket (orderbook_delta) for the BTC 15-minute Up/Down market (series KXBTC15M).

Kalshi sends only BIDS: `yes` bids and `no` bids. A YES ask at price p is a NO bid at 1-p, so we convert to the same
(bids, asks) shape as the Polymarket feed (UsBook, prices in dollars = probability of Up) to compare them directly.
A snapshot arrives first, then incremental deltas with a sequence number; a gap means a missed update, so we
reconnect and take a fresh snapshot rather than trust a book we know is wrong.
"""
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import requests
import websockets

from .brti import WS_URL, KalshiAuth
from .us_feed import UsBook

log = logging.getLogger(__name__)
REST = "https://api.elections.kalshi.com"
SERIES = "KXBTC15M"


@dataclass(frozen=True)
class KalshiMarket:
    ticker: str
    open_ts: float
    close_ts: float
    reference: float  # floor_strike: the price Up must match or beat (same number as Polymarket's priceToBeat)


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def parse_market(m: dict) -> KalshiMarket | None:
    try:
        if m.get("strike_type") != "greater_or_equal" or m.get("floor_strike") is None:
            return None
        return KalshiMarket(m["ticker"], _ts(m["open_time"]), _ts(m["close_time"]), float(m["floor_strike"]))
    except (KeyError, ValueError, TypeError):
        return None


def find_current(auth: KalshiAuth, now: float | None = None, session=None) -> KalshiMarket | None:
    """The open KXBTC15M market whose window contains `now` (else the soonest one still to close)."""
    s = session or requests
    now = now or time.time()
    path = "/trade-api/v2/markets"
    r = s.get(REST + path, params={"series_ticker": SERIES, "status": "open", "limit": 10},
              headers=auth.headers("GET", path), timeout=15)
    r.raise_for_status()
    ms = sorted((x for x in (parse_market(m) for m in r.json().get("markets", [])) if x and x.close_ts > now),
                key=lambda x: x.close_ts)
    return next((m for m in ms if m.open_ts <= now), ms[0] if ms else None)


class KalshiBook:
    """Mutable order book for one market: price -> quantity for YES bids and NO bids."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.yes: dict[float, float] = {}
        self.no: dict[float, float] = {}

    @staticmethod
    def _levels(raw) -> dict[float, float]:
        out = {}
        for p, q in raw or []:
            p, q = float(p), float(q)
            if q > 0:
                out[round(p, 4)] = q
        return out

    def snapshot(self, yes_levels, no_levels) -> None:
        self.yes, self.no = self._levels(yes_levels), self._levels(no_levels)

    def delta(self, side: str, price: float, dq: float) -> None:
        book = self.yes if side == "yes" else self.no
        p = round(float(price), 4)
        q = book.get(p, 0.0) + float(dq)
        if q > 1e-9:
            book[p] = q
        else:
            book.pop(p, None)

    def to_book(self, ts_ms: float | None = None) -> UsBook:
        bids = sorted(self.yes.items(), reverse=True)                       # YES bids, best first
        asks = sorted(((round(1 - p, 4), q) for p, q in self.no.items()))   # YES ask = 1 - NO bid, best first
        return UsBook(slug=self.ticker, bids=bids, asks=asks, exchange_ts=ts_ms / 1000 if ts_ms else None)


class SeqGap(Exception):
    pass


def apply_message(book: KalshiBook, raw: str | bytes, state: dict) -> bool:
    """Apply one websocket frame to the book. Returns True if the book changed. Raises SeqGap on a missed message."""
    try:
        m = json.loads(raw)
    except ValueError:
        return False
    kind = m.get("type")
    if kind not in ("orderbook_snapshot", "orderbook_delta"):
        if kind == "error":
            log.warning("kalshi error frame: %.200s", raw)
        return False
    msg = m.get("msg") or {}
    if msg.get("market_ticker") != book.ticker:
        return False
    seq = m.get("seq")
    if kind == "orderbook_snapshot":
        book.snapshot(msg.get("yes_dollars_fp"), msg.get("no_dollars_fp"))
        state["seq"], state["have_snapshot"] = seq, True
        return True
    if not state.get("have_snapshot"):
        return False                                     # a delta before the snapshot is meaningless
    if seq is not None and state.get("seq") is not None and seq != state["seq"] + 1:
        raise SeqGap(f"expected seq {state['seq'] + 1}, got {seq}")
    state["seq"] = seq
    book.delta(msg["side"], msg["price_dollars"], msg["delta_fp"])
    state["ts_ms"] = msg.get("ts_ms")
    return True


class KalshiBookFeed:
    """Background client that follows the current 15-minute market, rolling to the next one when it closes."""

    def __init__(self, auth: KalshiAuth, url: str = WS_URL):
        self.auth, self.url = auth, url
        self.market: KalshiMarket | None = None
        self._book: UsBook | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.on_update = None  # callback(UsBook) per update, on the feed thread
        self.stats = {"frames": 0, "updates": 0, "reconnects": 0, "gaps": 0}

    def book(self) -> UsBook | None:
        with self._lock:
            return self._book if self._connected else None

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="kalshi-feed").start()

    def stop(self) -> None:
        self._stop.set()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                mkt = await asyncio.to_thread(find_current, self.auth)
                if mkt is None:
                    await asyncio.sleep(3)
                    continue
                self.market = mkt
                await self._session(mkt)
                backoff = 1.0
            except SeqGap as e:
                self.stats["gaps"] += 1
                log.warning("Kalshi feed: %s -> resyncing", e)
            except Exception as e:
                self.stats["reconnects"] += 1
                log.warning("Kalshi feed error: %s (retry in %.0fs)", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _session(self, mkt: KalshiMarket) -> None:
        kb, state = KalshiBook(mkt.ticker), {}
        async with websockets.connect(self.url, additional_headers=self.auth.headers(), ping_interval=5, ping_timeout=5) as ws:
            await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                      "params": {"channels": ["orderbook_delta"], "market_tickers": [mkt.ticker]}}))
            log.info("Kalshi feed connected: %s (ref %.2f, closes in %.0fs)", mkt.ticker, mkt.reference, mkt.close_ts - time.time())
            try:
                while not self._stop.is_set():
                    if time.time() > mkt.close_ts + 2:          # this market is over: go find the next one
                        return
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    self.stats["frames"] += 1
                    if apply_message(kb, raw, state):
                        b = kb.to_book(state.get("ts_ms"))
                        with self._lock:
                            self._book, self._connected = b, True
                        self.stats["updates"] += 1
                        if self.on_update:
                            try:
                                self.on_update(b)
                            except Exception:
                                log.exception("on_update failed")
            finally:
                with self._lock:
                    self._connected, self._book = False, None    # never serve a book from a dead/closed session
