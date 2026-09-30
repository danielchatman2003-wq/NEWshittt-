"""Order placement for one Polymarket US market: LiveBroker (real REST) and PaperBroker (simulated).

Both expose: place(intent, price, qty) -> id|None, cancel(id), cancel_all(), open_orders(), position().
Prices are ALWAYS the YES price (docs: to buy NO at 0.83 send price 0.17).  position() is net YES
contracts: >0 long YES, <0 long NO.  All four intents reduce to buy/sell of YES for accounting.
"""
import itertools
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .us_feed import REST, UsAuth

log = logging.getLogger(__name__)

BUY_LONG, SELL_LONG = "ORDER_INTENT_BUY_LONG", "ORDER_INTENT_SELL_LONG"
BUY_SHORT, SELL_SHORT = "ORDER_INTENT_BUY_SHORT", "ORDER_INTENT_SELL_SHORT"
BUYS_YES = {BUY_LONG, SELL_SHORT}  # intents that add YES exposure


@dataclass(frozen=True)
class RestingOrder:
    id: str
    intent: str
    price: float
    qty: float


class UsApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


class RateLimiter:
    """Token bucket; the exchange allows 20 req/s per key, we stay well under."""

    def __init__(self, rps: float = 10.0, burst: int = 5):
        self.rps, self.burst, self._tokens, self._t = rps, burst, float(burst), time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rps)
            self._t = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self.rps
                time.sleep(wait)
                self._tokens, self._t = 0.0, time.monotonic()
            else:
                self._tokens -= 1


def order_body(slug: str, intent: str, price: float, qty: float, tick_decimals: int = 2,
               expire_at: float | None = None) -> dict:
    """Maker-only limit order. participateDontInitiate => rejected instead of ever taking (and paying fees).
    With expire_at (unix s) the order is GTD: if the bot dies, the exchange removes it by itself."""
    body = {
        "marketSlug": slug,
        "type": "ORDER_TYPE_LIMIT",
        "price": {"value": f"{price:.{tick_decimals}f}", "currency": "USD"},
        "quantity": qty,
        "tif": "TIME_IN_FORCE_GOOD_TILL_CANCEL",
        "intent": intent,
        "participateDontInitiate": True,
        "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
    }
    if expire_at is not None:
        body["tif"] = "TIME_IN_FORCE_GOOD_TILL_DATE"
        body["goodTillTime"] = datetime.fromtimestamp(expire_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return body


class UsRest:
    def __init__(self, auth: UsAuth, rps: float = 10.0, session: requests.Session | None = None):
        self.auth, self.limiter, self.s = auth, RateLimiter(rps), session or requests.Session()

    def call(self, method: str, path: str, params=None, body=None, priority: bool = False) -> dict:
        for attempt in (0, 1):
            if not priority:  # cancels skip the queue: a pull must never wait behind polling
                self.limiter.acquire()
            data = json.dumps(body, separators=(",", ":")) if body is not None else None
            headers = {**self.auth.headers(method, path), "Content-Type": "application/json"}  # signs path only
            r = self.s.request(method, REST + path, params=params, data=data, headers=headers, timeout=5)
            if r.status_code == 429 and attempt == 0:
                time.sleep(1.0)  # docs: stop, wait >= 1s, retry
                continue
            if not r.ok:
                raise UsApiError(r.status_code, r.text[:300])
            return r.json() if r.content else {}
        raise UsApiError(429, "rate limited")

    def preview(self, body: dict) -> dict:
        """Validate an order against the real API WITHOUT placing it (documented as side-effect free)."""
        return self.call("POST", "/v1/order/preview", body={"request": body})


class LiveBroker:
    """Real orders. Reads (open orders, position, buying power) are cached ~1s and our own places/cancels
    update the cache immediately, so the 4Hz quoting loop costs ~3 requests/s, not 12."""

    LIST_LAG_GRACE = 4.0  # seconds a new order may be missing from the exchange's list before we believe it's gone

    def __init__(self, rest: UsRest, slug: str, expire_at: float | None = None, refresh: float = 1.0):
        self.rest, self.slug, self.expire_at, self.refresh = rest, slug, expire_at, refresh
        self._orders: dict[str, RestingOrder] = {}
        self._placed_at: dict[str, float] = {}  # order id -> monotonic time we placed it
        self._pos, self._bp = 0.0, None
        self._pending = (0.0, 0.0)  # (assumed fill qty not yet in the position, when seen)
        self._last = 0.0
        self._warned = 0.0
        self.fault: str | None = None  # set if the exchange's position contradicts our fills (e.g. sign flipped)

    def _refresh(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._last < self.refresh:
            return
        j = self.rest.call("GET", "/v1/orders/open", params={"slugs": self.slug})
        fresh = {o["id"]: RestingOrder(o["id"], o.get("intent", ""), float(o["price"]["value"]),
                                       float(o.get("leavesQuantity", o.get("quantity", 0))))
                 for o in j.get("orders", [])}
        pj = self.rest.call("GET", "/v1/portfolio/positions", params={"market": self.slug})
        p = (pj.get("positions") or {}).get(self.slug)
        pos = float(p["netPositionDecimal"]) if p else 0.0
        try:
            bj = self.rest.call("GET", "/v1/account/balances")
            self._bp = float(bj["balances"][0]["buyingPower"])
        except Exception:
            pass  # keep the last known value
        # A just-placed order can take a moment to show up in the exchange's open-orders list. Dropping it here
        # would make the bot think nothing is resting and place a DUPLICATE, so carry young orders forward.
        now = time.monotonic()
        for i, o in self._orders.items():
            if i not in fresh and now - self._placed_at.get(i, 0.0) < self.LIST_LAG_GRACE:
                fresh[i] = o
        gone = [o for i, o in self._orders.items()
                if i not in fresh and now - self._placed_at.get(i, 0.0) >= self.LIST_LAG_GRACE]  # filled/expired
        expected = sum(o.qty if o.intent in BUYS_YES else -o.qty for o in gone)
        delta = pos - self._pos
        if gone and abs(delta) > 1e-9 and expected * delta < 0:
            self.fault = (f"position moved {delta:+g} but the orders that vanished imply {expected:+g}: "
                          "position sign convention is not what we assumed - STOPPING")
        if gone and abs(delta) < 1e-9 and expected:
            self._pending = (expected, now)   # an order vanished but the position hasn't moved yet: assume it filled
        elif abs(delta) > 1e-9:
            self._pending = (0.0, 0.0)        # position caught up
        self._orders, self._pos, self._last = fresh, pos, time.monotonic()

    def _cost(self, intent: str, price: float, qty: float) -> float:
        if intent == BUY_LONG:
            return price * qty
        if intent == BUY_SHORT:
            return (1 - price) * qty
        return 0.0  # closing orders need no new cash

    def place(self, intent: str, price: float, qty: float) -> str | None:
        cost = self._cost(intent, price, qty)
        if self._bp is not None and cost > self._bp:
            if time.monotonic() - self._warned > 30:
                log.warning("skip %s %.2f x%g: needs $%.2f, buying power $%.2f", intent[13:], price, qty, cost, self._bp)
                self._warned = time.monotonic()
            return None
        try:
            resp = self.rest.call("POST", "/v1/orders", body=order_body(self.slug, intent, price, qty, expire_at=self.expire_at))
        except UsApiError as e:  # a rejected maker-only quote (would cross) must not kill the loop
            log.warning("place %s %.2f x%s rejected: %s", intent[13:], price, qty, e)
            return None
        oid = resp.get("id")
        if oid:
            self._orders[oid] = RestingOrder(oid, intent, price, qty)
            self._placed_at[oid] = time.monotonic()
            if self._bp is not None:
                self._bp -= cost
            log.info("[LIVE] PLACE %-10s %.2f x%g  id=%s", intent[13:], price, qty, oid)
        return oid

    def cancel(self, order_id: str) -> None:
        self._orders.pop(order_id, None)
        self._placed_at.pop(order_id, None)
        self.rest.call("POST", f"/v1/order/{order_id}/cancel", body={"marketSlug": self.slug}, priority=True)

    def cancel_all(self) -> None:
        self._orders.clear()
        self._placed_at.clear()
        self.rest.call("POST", "/v1/orders/open/cancel", body={"slugs": [self.slug]}, priority=True)
        self._last = 0.0  # re-read truth next time

    def open_orders(self) -> list[RestingOrder]:
        self._refresh()
        return list(self._orders.values())

    def position(self) -> float:
        """Net YES contracts. Conservative: a just-vanished order counts as filled until the position confirms
        (for ~5s), so the position cap can't be overshot by a laggy read."""
        self._refresh()
        qty, t = self._pending
        return self._pos + (qty if qty and time.monotonic() - t < 5.0 else 0.0)


class PaperBroker:
    """Simulated fills against the live book. A resting bid fills when the market trades DOWN through
    it (best bid drops below our price); an ask fills when it trades UP through it. Deliberately
    conservative: it captures adverse selection (the real risk) and misses benign fills at the touch."""

    def __init__(self, maker_rebate: float = 0.0125):
        self.rebate = maker_rebate
        self._orders: dict[str, RestingOrder] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.pos = 0.0
        self.cash = 0.0
        self.fills: list[tuple[str, float, float]] = []  # (intent, price, qty)

    def place(self, intent: str, price: float, qty: float) -> str:
        with self._lock:
            oid = f"paper-{next(self._ids)}"
            self._orders[oid] = RestingOrder(oid, intent, price, qty)
        log.info("[paper] PLACE %-9s %.2f x%g", _short(intent), price, qty)
        return oid

    def cancel(self, order_id: str) -> None:
        with self._lock:
            self._orders.pop(order_id, None)

    def cancel_all(self) -> None:
        with self._lock:
            n = len(self._orders)
            self._orders.clear()
        if n:
            log.info("[paper] CANCEL ALL (%d)", n)

    def open_orders(self) -> list[RestingOrder]:
        with self._lock:
            return list(self._orders.values())

    def position(self) -> float:
        return self.pos

    def on_book(self, best_bid: float | None, best_ask: float | None) -> None:
        with self._lock:
            for o in list(self._orders.values()):
                buys = o.intent in BUYS_YES
                hit = (buys and best_bid is not None and best_bid < o.price) or (
                    not buys and best_ask is not None and best_ask > o.price)
                if hit:
                    self._fill(o)

    def _fill(self, o: RestingOrder) -> None:
        del self._orders[o.id]
        buys = o.intent in BUYS_YES
        self.pos += o.qty if buys else -o.qty
        self.cash += (-1 if buys else 1) * o.price * o.qty + self.rebate * o.price * (1 - o.price) * o.qty
        self.fills.append((o.intent, o.price, o.qty))
        log.info("[paper] FILL  %-9s %.2f x%g  -> pos %+g", _short(o.intent), o.price, o.qty, self.pos)

    def settle(self, up: bool) -> float:
        """Close the window: YES pays $1 if Up. Returns realised P&L (cash incl. rebates) and resets."""
        with self._lock:
            pnl = self.cash + self.pos * (1.0 if up else 0.0)
            self._orders.clear()
            self.pos = self.cash = 0.0
            return pnl


def _short(intent: str) -> str:
    return intent.replace("ORDER_INTENT_", "")
