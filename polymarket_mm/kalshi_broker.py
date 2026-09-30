"""Live order placement on Kalshi (V2 orders API) for ONE market, same interface as the Polymarket LiveBroker.

Kalshi V2 orders live in the YES book: side "bid" = buy YES (Up) exposure, "ask" = sell YES = buy NO (Down).
Prices are YES prices in fixed-point dollars ("0.3000") and counts are fixed-point strings ("1.00"), exactly like
the Polymarket code here, so the same quote logic drives both. Position sign: positive = YES (Up), negative = NO.
"""
import logging
import time
import uuid

import requests

from .brti import KalshiAuth
from .us_broker import BUY_LONG, BUY_SHORT, BUYS_YES, SELL_LONG, SELL_SHORT, RateLimiter, RestingOrder

log = logging.getLogger(__name__)
HOST = "https://api.elections.kalshi.com"
API = "/trade-api/v2"


class KalshiApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


class KalshiRest:
    def __init__(self, auth: KalshiAuth, rps: float = 8.0, session=None):
        self.auth, self.limiter, self.s = auth, RateLimiter(rps), session or requests.Session()

    def call(self, method: str, path: str, params=None, body=None, priority: bool = False) -> dict:
        full = API + path
        for attempt in (0, 1):
            if not priority:
                self.limiter.acquire()
            tries = 3 if (method in ("GET", "DELETE") or priority) else 1        # never auto-retry order creation
            for k in range(tries):
                try:
                    r = self.s.request(method, HOST + full, params=params, json=body, headers=self.auth.headers(method, full), timeout=6)
                    break
                except (requests.ConnectionError, requests.Timeout):
                    if k == tries - 1:
                        raise
                    time.sleep(0.3 * (k + 1))
            if r.status_code == 429 and attempt == 0:
                time.sleep(1.0)
                continue
            if not r.ok:
                raise KalshiApiError(r.status_code, r.text[:300])
            return r.json() if r.content else {}
        raise KalshiApiError(429, "rate limited")


def order_body(ticker: str, intent: str, price: float, qty: float, expire_at: float | None = None, take: bool = False) -> dict:
    """Maker-only (post_only) GTC/GTD limit order, or an IOC taker order when take=True. Price = YES price."""
    body = {
        "ticker": ticker,
        "side": "bid" if intent in BUYS_YES else "ask",
        "count": f"{qty:.2f}",
        "price": f"{price:.4f}",
        "time_in_force": "immediate_or_cancel" if take else "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": not take,
        "client_order_id": str(uuid.uuid4()),
        "reduce_only": intent in (SELL_LONG, SELL_SHORT),       # an exit may only reduce an existing position
    }
    if expire_at is not None and not take:
        body["expiration_time"] = int(expire_at)
    return body


def _intent_of(book_side: str) -> str:
    return BUY_LONG if book_side == "bid" else BUY_SHORT         # bid = Up exposure, ask = Down exposure


class KalshiLiveBroker:
    def __init__(self, rest: KalshiRest, ticker: str, expire_at: float | None = None, refresh: float = 1.0):
        self.rest, self.ticker, self.expire_at, self.refresh = rest, ticker, expire_at, refresh
        self._orders: dict[str, RestingOrder] = {}
        self._placed_at: dict[str, float] = {}
        self._pos, self._last, self._bp = 0.0, 0.0, None
        self._pending = (0.0, 0.0)
        self.fault: str | None = None

    LIST_LAG_GRACE = 4.0

    def _refresh(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._last < self.refresh:
            return
        j = self.rest.call("GET", "/portfolio/orders", params={"ticker": self.ticker, "status": "resting"})
        fresh = {o["order_id"]: RestingOrder(o["order_id"], _intent_of(o.get("book_side", "bid")), float(o["yes_price_dollars"]),
                                             float(o.get("remaining_count_fp", o.get("initial_count_fp", 0))))
                 for o in j.get("orders", [])}
        pj = self.rest.call("GET", "/portfolio/positions", params={"ticker": self.ticker})
        mp = next((x for x in pj.get("market_positions", []) if x.get("ticker") == self.ticker), None)
        pos = float(mp["position_fp"]) if mp else 0.0
        try:
            self._bp = float(self.rest.call("GET", "/portfolio/balance")["balance"]) / 100.0
        except Exception:
            pass
        now = time.monotonic()
        for i, o in self._orders.items():                      # a just-placed order may not be listed yet: keep it
            if i not in fresh and now - self._placed_at.get(i, 0.0) < self.LIST_LAG_GRACE:
                fresh[i] = o
        gone = [o for i, o in self._orders.items() if i not in fresh]
        expected = sum(o.qty if o.intent in BUYS_YES else -o.qty for o in gone)
        delta = pos - self._pos
        if gone and abs(delta) > 1e-9 and expected * delta < 0:
            self.fault = f"Kalshi position moved {delta:+g} but vanished orders imply {expected:+g}: sign convention wrong - STOPPING"
        if gone and abs(delta) < 1e-9 and expected:
            self._pending = (expected, now)
        elif abs(delta) > 1e-9:
            self._pending = (0.0, 0.0)
        self._orders, self._pos, self._last = fresh, pos, now

    def place(self, intent: str, price: float, qty: float) -> str | None:
        cost = price * qty if intent in BUYS_YES else (1 - price) * qty
        if self._bp is not None and cost > self._bp:
            log.warning("[Kalshi] skip %s %.4f: needs $%.2f, balance $%.2f", intent[13:], price, cost, self._bp)
            return None
        try:
            resp = self.rest.call("POST", "/portfolio/events/orders", body=order_body(self.ticker, intent, price, qty, self.expire_at))
        except KalshiApiError as e:
            log.warning("[Kalshi] place %s %.4f rejected: %s", intent[13:], price, e)
            return None
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning("[Kalshi] place network error (%s): re-checking open orders", type(e).__name__)
            self._last = 0.0
            return None
        oid = resp.get("order_id")
        if oid:
            self._orders[oid] = RestingOrder(oid, intent, price, qty)
            self._placed_at[oid] = time.monotonic()
            if self._bp is not None:
                self._bp -= cost
            log.info("[Kalshi LIVE] PLACE %-10s %.4f x%g id=%s", intent[13:], price, qty, oid[:8])
        return oid

    def take(self, intent: str, price: float, qty: float) -> bool:
        try:
            resp = self.rest.call("POST", "/portfolio/events/orders", body=order_body(self.ticker, intent, price, qty, take=True))
        except (KalshiApiError, requests.ConnectionError, requests.Timeout) as e:
            log.warning("[Kalshi] TAKE %s %.4f failed: %s", intent[13:], price, str(e)[:120])
            self._last = 0.0
            return False
        filled = float(resp.get("fill_count") or 0)
        log.info("[Kalshi LIVE] TAKE  %-10s limit %.4f x%g  filled=%g", intent[13:], price, qty, filled)
        self._last = 0.0
        if filled <= 1e-9:
            return False                                   # IOC found nothing at our limit: do NOT pretend it filled
        self._pending = (filled if intent in BUYS_YES else -filled, time.monotonic())
        return True

    def cancel(self, order_id: str) -> None:
        self._orders.pop(order_id, None)
        self._placed_at.pop(order_id, None)
        self.rest.call("DELETE", f"/portfolio/events/orders/{order_id}", params={"market_ticker": self.ticker}, priority=True)

    def cancel_all(self) -> None:
        self._last = 0.0
        self._refresh(force=True)
        for oid in list(self._orders):
            try:
                self.cancel(oid)
            except KalshiApiError as e:
                log.warning("[Kalshi] cancel %s failed: %s", oid[:8], e)
        self._orders.clear()
        self._placed_at.clear()
        self._last = 0.0

    def open_orders(self) -> list[RestingOrder]:
        self._refresh()
        return [RestingOrder(o.id, o.intent, o.price, o.qty, self._placed_at.get(o.id, 0.0)) for o in self._orders.values()]

    def position(self) -> float:
        self._refresh()
        qty, t = self._pending
        return self._pos + (qty if qty and time.monotonic() - t < 5.0 else 0.0)
