"""Adverse-selection defences used by professional market makers, in pure testable pieces.

 * depth imbalance + microprice : which side of the book is thin / about to give way
 * TradeFlow                    : who has been hitting bids vs lifting asks over the last few seconds
 * ToxicityGate                 : trip on pressure, re-arm only after it clears (hysteresis) + a cooldown
 * MarkoutTracker               : what the price did AFTER each of our fills -> widen / pause when we keep getting picked off

Sign convention everywhere: pressure > 0 means UP pressure (buyers dominate), < 0 means DOWN pressure (sellers dominate).
Down pressure endangers our BIDS (an Up bid gets hit as price falls); up pressure endangers our ASKS (= Down bids).
"""
import json
from collections import deque
from datetime import datetime


def depth_imbalance(bk, levels: int = 3) -> float:
    """(bid size - ask size) / total over the top `levels` of each side, in [-1, 1]. > 0: bid-heavy (price leans up)."""
    b = sum(q for _, q in bk.bids[:levels])
    a = sum(q for _, q in bk.asks[:levels])
    return 0.0 if b + a <= 0 else (b - a) / (b + a)


def micro_skew(bk) -> float:
    """(microprice - mid) / (spread/2) in [-1, 1]. The microprice weights each side's price by the OTHER side's size, so it
    sits closer to the thin side; > 0 means the ask side is the thin one (price leans up)."""
    if bk.best_bid is None or bk.best_ask is None:
        return 0.0
    qb, qa = bk.bids[0][1], bk.asks[0][1]
    spread = bk.best_ask - bk.best_bid
    if qb + qa <= 0 or spread <= 0:
        return 0.0
    micro = (bk.best_bid * qa + bk.best_ask * qb) / (qb + qa)
    return max(-1.0, min(1.0, (micro - (bk.best_bid + bk.best_ask) / 2) / (spread / 2)))


def parse_trade(raw) -> tuple[float, float, int, float | None] | None:
    """Polymarket US trade frame -> (price, qty, taker_sign, ts). taker_sign: +1 a buyer lifted the ask, -1 a seller hit the
    bid. The frame names the MAKER's side in the YES book: maker BUY -> the taker sold; maker SELL -> the taker bought."""
    try:
        t = (json.loads(raw) if isinstance(raw, (str, bytes)) else raw).get("trade")
        if not t:
            return None
        px, qty = float(t["price"]["value"]), float(t["quantity"]["value"])
        side = (t.get("maker") or {}).get("side", "")
        sign = -1 if side.endswith("BUY") else 1 if side.endswith("SELL") else 0
        ts = datetime.fromisoformat(t["tradeTime"].replace("Z", "+00:00")[:26] + "+00:00").timestamp() if t.get("tradeTime") else None
        return px, qty, sign, ts
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


class TradeFlow:
    """Rolling signed trade volume. imbalance() in [-1, 1]: +1 all buyers lifting, -1 all sellers hitting."""

    def __init__(self, window: float = 10.0, min_volume: float = 5.0):
        self.window, self.min_volume = window, min_volume
        self._t: deque = deque()

    def add(self, now: float, qty: float, taker_sign: int) -> None:
        self._t.append((now, taker_sign * qty, qty))

    def imbalance(self, now: float) -> float:
        while self._t and self._t[0][0] < now - self.window:
            self._t.popleft()
        total = sum(q for _, _, q in self._t)
        return 0.0 if total < self.min_volume else sum(s for _, s, _ in self._t) / total


def pressure(bk, flow_imbalance: float) -> float:
    """Combined up(+)/down(-) pressure in [-1, 1]: average of depth imbalance, microprice skew and trade flow."""
    return (depth_imbalance(bk) + micro_skew(bk) + flow_imbalance) / 3.0


class ToxicityGate:
    """Trip when pressure reaches +/-trip; clear only when it falls back inside +/-rearm (hysteresis); then stay blocked
    for `cooldown` more seconds. block_up_bids: down pressure, so don't rest UP bids. block_down_bids: up pressure."""

    def __init__(self, trip: float = 0.45, rearm: float = 0.25, cooldown: float = 3.0):
        self.trip, self.rearm, self.cooldown = trip, rearm, cooldown
        self.down = self.up = False
        self._cd_down = self._cd_up = 0.0

    def update(self, now: float, p: float) -> tuple[bool, bool, bool, bool]:
        """Returns (block_up_bids, block_down_bids, newly_down, newly_up)."""
        newly_down = newly_up = False
        if not self.down and p <= -self.trip:
            self.down, newly_down = True, True
        elif self.down and p > -self.rearm:
            self.down, self._cd_down = False, now + self.cooldown
        if not self.up and p >= self.trip:
            self.up, newly_up = True, True
        elif self.up and p < self.rearm:
            self.up, self._cd_up = False, now + self.cooldown
        return (self.down or now < self._cd_down, self.up or now < self._cd_up, newly_down, newly_up)


class MarkoutTracker:
    """After each fill, compare the mid `horizon` seconds later with our fill price (direction +1: we bought YES exposure,
    -1: we sold it). EWMA of those markouts < 0 means we are systematically picked off."""

    def __init__(self, horizon: float = 10.0, alpha: float = 0.3):
        self.horizon, self.alpha = horizon, alpha
        self._pending: deque = deque()
        self.ewma: float | None = None
        self.n = 0

    def add(self, now: float, direction: int, price: float) -> None:
        self._pending.append((now + self.horizon, direction, price))

    def update(self, now: float, mid: float | None) -> list[float]:
        out = []
        while self._pending and self._pending[0][0] <= now and mid is not None:
            _, d, p = self._pending.popleft()
            m = d * (mid - p)
            self.ewma = m if self.ewma is None else (1 - self.alpha) * self.ewma + self.alpha * m
            self.n += 1
            out.append(m)
        return out
