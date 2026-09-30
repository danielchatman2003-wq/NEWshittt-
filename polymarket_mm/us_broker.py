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


def order_body(slug: str, intent: str, price: float, qty: float, tick_decimals: int = 2) -> dict:
    """Maker-only GTC limit order. participateDontInitiate => rejected instead of ever taking (and paying fees)."""
    return {
        "marketSlug": slug,
        "type": "ORDER_TYPE_LIMIT",
        "price": {"value": f"{price:.{tick_decimals}f}", "currency": "USD"},
        "quantity": qty,
        "tif": "TIME_IN_FORCE_GOOD_TILL_CANCEL",
        "intent": intent,
        "participateDontInitiate": True,
        "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
    }


class UsRest:
    def __init__(self, auth: UsAuth, rps: float = 10.0, session: requests.Session | None = None):
        self.auth, self.limiter, self.s = auth, RateLimiter(rps), session or requests.Session()

    def call(self, method: str, path: str, params=None, body=None) -> dict:
        for attempt in (0, 1):
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
    def __init__(self, rest: UsRest, slug: str):
        self.rest, self.slug = rest, slug

    def place(self, intent: str, price: float, qty: float) -> str | None:
        try:
            resp = self.rest.call("POST", "/v1/orders", body=order_body(self.slug, intent, price, qty))
            return resp.get("id")
        except UsApiError as e:  # a rejected maker-only quote (would cross) must not kill the loop
            log.warning("place %s %.2f x%s rejected: %s", intent, price, qty, e)
            return None

    def cancel(self, order_id: str) -> None:
        self.rest.call("POST", f"/v1/order/{order_id}/cancel", body={})

    def cancel_all(self) -> None:
        self.rest.call("POST", "/v1/orders/open/cancel", body={"slugs": [self.slug]})

    def open_orders(self) -> list[RestingOrder]:
        j = self.rest.call("GET", "/v1/orders/open", params={"slugs": self.slug})
        return [RestingOrder(o["id"], o.get("intent", ""), float(o["price"]["value"]),
                             float(o.get("leavesQuantity", o.get("quantity", 0)))) for o in j.get("orders", [])]

    def position(self) -> float:
        j = self.rest.call("GET", "/v1/portfolio/positions", params={"market": self.slug})
        p = (j.get("positions") or {}).get(self.slug)
        return float(p["netPositionDecimal"]) if p else 0.0


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
