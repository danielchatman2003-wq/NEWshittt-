"""Polymarket US markets websocket: live full order books, event-driven (no polling).

wss://api.polymarket.us/v1/ws/markets, SUBSCRIPTION_TYPE_MARKET_DATA. Every frame is a full
book snapshot (the docs describe no deltas), so each update simply replaces the stored book.
`responsesDebounced` is False so we get every update instead of a batched one.
"""
import asyncio
import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import websockets
from cryptography.hazmat.primitives.asymmetric import ed25519

log = logging.getLogger(__name__)
WS_URL = "wss://api.polymarket.us/v1/ws/markets"
WS_PATH = "/v1/ws/markets"
REST = "https://api.polymarket.us"
SUB_MARKET_DATA = "SUBSCRIPTION_TYPE_MARKET_DATA"


class UsAuth:
    def __init__(self, key_id: str, secret_b64: str):
        self.key_id = key_id
        # per the docs: base64-decode the secret and use the first 32 bytes as the Ed25519 seed
        self._key = ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(secret_b64)[:32])

    @classmethod
    def from_env(cls) -> "UsAuth":
        kid = os.getenv("POLYMARKET_US_KEY_ID", "").strip()
        sec = os.getenv("POLYMARKET_US_SECRET_KEY", "").strip()
        if not kid or not sec:
            raise SystemExit("Set POLYMARKET_US_KEY_ID and POLYMARKET_US_SECRET_KEY in .env")
        return cls(kid, sec)

    def headers(self, method: str = "GET", path: str = WS_PATH) -> dict:
        ts = str(int(time.time() * 1000))  # must be within 30s of server time
        sig = base64.b64encode(self._key.sign(f"{ts}{method}{path}".encode())).decode()
        return {"X-PM-Access-Key": self.key_id, "X-PM-Timestamp": ts, "X-PM-Signature": sig}


def _px(o) -> float | None:
    """Prices arrive as {"value": "0.555", "currency": "USD"}; tolerate a bare number too."""
    try:
        return float(o["value"] if isinstance(o, dict) else o)
    except (TypeError, ValueError, KeyError):
        return None


def _levels(raw, reverse: bool) -> list[tuple[float, float]]:
    out = []
    for lv in raw or []:
        p, q = _px(lv.get("px")), _px(lv.get("qty"))
        if p is not None and q:
            out.append((p, q))
    return sorted(out, reverse=reverse)


@dataclass(frozen=True)
class UsBook:
    slug: str
    bids: list[tuple[float, float]]  # (price, qty), best (highest) first
    asks: list[tuple[float, float]]  # best (lowest) first
    state: str | None = None
    last_trade: float | None = None
    exchange_ts: float | None = None  # transactTime, unix seconds
    local_ts: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        return (b + a) / 2 if b is not None and a is not None else None

    @property
    def spread(self) -> float | None:
        b, a = self.best_bid, self.best_ask
        return a - b if b is not None and a is not None else None


def _iso_to_ts(s) -> float | None:
    if not s:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_frame(raw: str | bytes) -> UsBook | None:
    """Parse a marketData frame into a UsBook; None for heartbeats, acks, trades, junk."""
    try:
        m = json.loads(raw)
    except ValueError:
        return None
    md = m.get("marketData") if isinstance(m, dict) else None
    if not md or not md.get("marketSlug"):
        return None
    stats = md.get("stats") or {}
    return UsBook(
        slug=md["marketSlug"],
        bids=_levels(md.get("bids"), reverse=True),
        asks=_levels(md.get("offers"), reverse=False),
        state=md.get("state"),
        last_trade=_px(stats.get("lastTradePx")),
        exchange_ts=_iso_to_ts(md.get("transactTime")),
    )


class UsBookFeed:
    """Background client. `book(slug)` from any thread, or set `on_update(book)` for push.

    Health = the socket being alive (protocol pings), NOT frame silence: a quiet market
    legitimately sends nothing for minutes, and a book only changes when the market does.
    """

    def __init__(self, auth: UsAuth, slugs: list[str], url: str = WS_URL, ping_every: float = 5.0):
        self.auth, self.slugs, self.url, self.ping_every = auth, list(slugs), url, ping_every
        self._books: dict[str, UsBook] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self.on_update = None  # callback(UsBook) on the feed thread, per update
        self.on_trade = None   # optional callback(raw_frame) for trade frames; if set we also subscribe to the trade stream
        self.stats = {"frames": 0, "updates": 0, "reconnects": 0}

    def book(self, slug: str) -> UsBook | None:
        """Latest book, or None if we've never had one or the connection is down (book may be stale)."""
        with self._lock:
            b = self._books.get(slug) if self._connected else None
        return b

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="us-feed").start()

    def stop(self) -> None:
        self._stop.set()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 1.0
            except Exception as e:
                self.stats["reconnects"] += 1
                log.warning("US feed error: %s (reconnecting in %.0fs)", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _session(self) -> None:
        async with websockets.connect(self.url, additional_headers=self.auth.headers(), ping_interval=self.ping_every, ping_timeout=self.ping_every) as ws:
            await ws.send(json.dumps({"subscribe": {
                "requestId": f"mm-{int(time.time())}",
                "subscriptionType": SUB_MARKET_DATA,
                "marketSlugs": self.slugs,
                "responsesDebounced": False,
            }}))
            if self.on_trade:
                await ws.send(json.dumps({"subscribe": {
                    "requestId": f"tr-{int(time.time())}",
                    "subscriptionType": "SUBSCRIPTION_TYPE_TRADE",
                    "marketSlugs": self.slugs,
                    "responsesDebounced": False,
                }}))
            log.info("US feed connected, subscribed to %d markets", len(self.slugs))
            self._connected = True
            try:
                await self._read(ws)
            finally:
                with self._lock:  # never serve a pre-disconnect book as current after a reconnect
                    self._connected = False
                    self._books.clear()

    async def _read(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            self.stats["frames"] += 1
            if self.on_trade and b'"trade"' in (raw if isinstance(raw, bytes) else raw.encode()):
                try:
                    self.on_trade(raw)
                except Exception:
                    log.exception("on_trade failed")
                continue
            book = parse_frame(raw)
            if book is None:
                log.debug("non-book frame: %.300s", raw)
                continue
            with self._lock:
                self._books[book.slug] = book
            self.stats["updates"] += 1
            if self.on_update:
                try:
                    self.on_update(book)
                except Exception:
                    log.exception("on_update callback failed")
