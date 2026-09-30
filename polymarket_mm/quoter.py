"""Pure quoting logic: given a book midpoint and inventory, decide what to rest.

A binary market has two tokens, YES and NO, with price(NO) = 1 - price(YES).
We quote both sides of the YES market without ever needing to short:

  bid side (want more YES exposure): SELL NO @ 1-bid if we hold NO, else BUY YES @ bid
  ask side (want less YES exposure): SELL YES @ ask  if we hold YES, else BUY NO @ 1-ask

So inventory is unwound first and new exposure is only opened when flat on that side.
"""
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from .config import Config


@dataclass(frozen=True)
class Quote:
    token_id: str
    side: str  # "BUY" | "SELL"
    price: float
    size: float


def _d(x) -> Decimal:
    return Decimal(str(x))


def _floor(x: Decimal, tick: Decimal) -> Decimal:
    return (x / tick).to_integral_value(ROUND_FLOOR) * tick


def _ceil(x: Decimal, tick: Decimal) -> Decimal:
    return (x / tick).to_integral_value(ROUND_CEILING) * tick


def compute_quotes(
    *,
    yes_token: str,
    no_token: str,
    mid: float,
    tick: float,
    min_size: float,
    yes_pos: float,
    no_pos: float,
    cfg: Config,
    max_half_spread: float | None = None,
) -> list[Quote]:
    """Return the quotes we want resting for one market (possibly empty)."""
    t = _d(tick)
    half = _d(cfg.half_spread)
    if max_half_spread is not None:  # stay inside the liquidity-rewards spread
        half = min(half, _d(max_half_spread))
    half = max(half, t)

    # Shift the mid against our net inventory: long YES -> quote lower to sell, buy less.
    net = yes_pos - no_pos
    skew = max(-1.0, min(1.0, net / cfg.max_position)) if cfg.max_position > 0 else 0.0
    reservation = _d(mid) - _d(cfg.skew) * _d(skew)

    bid = _floor(reservation - half, t)
    ask = _ceil(reservation + half, t)
    if bid < t or ask > 1 - t or bid >= ask:
        return []  # no room to quote safely at this price/tick

    size = max(cfg.quote_size, min_size)
    quotes: list[Quote] = []

    # bid side
    if no_pos >= min_size:
        quotes.append(Quote(no_token, "SELL", float(1 - bid), min(size, no_pos)))
    else:
        room = cfg.max_position - yes_pos
        if room >= min_size:
            quotes.append(Quote(yes_token, "BUY", float(bid), min(size, room)))

    # ask side
    if yes_pos >= min_size:
        quotes.append(Quote(yes_token, "SELL", float(ask), min(size, yes_pos)))
    else:
        room = cfg.max_position - no_pos
        if room >= min_size:
            quotes.append(Quote(no_token, "BUY", float(1 - ask), min(size, room)))

    return quotes
