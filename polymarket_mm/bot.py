import logging
import threading
import time

from .broker import DryRunBroker
from .config import Config
from .markets import Market, select_markets
from .quoter import Quote, compute_quotes

log = logging.getLogger(__name__)
RESELECT_SECONDS = 1800
REWARD_SPREAD_MARGIN = 0.9  # stay comfortably inside the rewards spread


def best_prices(book) -> tuple[float | None, float | None]:
    """Best bid/ask regardless of how the API orders the levels."""
    bids = [float(b.price) for b in book.bids or []]
    asks = [float(a.price) for a in book.asks or []]
    return (max(bids) if bids else None, min(asks) if asks else None)


class MarketMaker:
    def __init__(self, cfg: Config, clob, broker, markets: list[Market] | None = None):
        self.cfg, self.clob, self.broker = cfg, clob, broker
        self.markets = markets if markets is not None else []
        self.stop = threading.Event()
        self._prev_mid: dict[str, float] = {}
        self._errors = 0

    # -- one market ---------------------------------------------------------
    def _place(self, q: Quote, m: Market) -> None:
        if isinstance(self.broker, DryRunBroker):
            self.broker.place(q, m.condition_id)
        else:
            self.broker.place(q)
            log.info("PLACE %s %.2f @ %.3f  %s", q.side, q.size, q.price, m.question[:50])

    def _matches(self, o: dict, q: Quote, tick: float) -> bool:
        return (
            o["asset_id"] == q.token_id
            and o["side"] == q.side
            and abs(o["price"] - q.price) < tick / 2
            and abs(o["size"] - q.size) <= max(1.0, 0.25 * q.size)
        )

    def _quote_market(self, m: Market, notional_left: float) -> float:
        """Reconcile resting orders for one market. Returns BUY notional committed."""
        book = self.clob.get_order_book(m.yes_token)
        bid, ask = best_prices(book)
        if bid is None or ask is None or ask - bid > self.cfg.max_book_spread:
            log.info("pulling quotes, thin/wide book: %s", m.question[:50])
            self.broker.cancel_market(m.condition_id)
            return 0.0

        mid = (bid + ask) / 2
        prev = self._prev_mid.get(m.condition_id)
        self._prev_mid[m.condition_id] = mid
        if prev is not None and abs(mid - prev) > self.cfg.move_pause:
            log.info("mid jumped %.3f -> %.3f, pulling quotes: %s", prev, mid, m.question[:50])
            self.broker.cancel_market(m.condition_id)
            return 0.0

        tick = float(book.tick_size or m.tick)
        quotes = compute_quotes(
            yes_token=m.yes_token,
            no_token=m.no_token,
            mid=mid,
            tick=tick,
            min_size=float(book.min_order_size or m.min_size),
            yes_pos=self.broker.position(m.yes_token),
            no_pos=self.broker.position(m.no_token),
            cfg=self.cfg,
            max_half_spread=m.rewards_max_spread * REWARD_SPREAD_MARGIN if m.rewards_max_spread else None,
        )

        committed, wanted = 0.0, []
        for q in quotes:
            cost = q.price * q.size if q.side == "BUY" else 0.0
            if committed + cost > notional_left:
                log.info("notional cap reached, dropping %s quote: %s", q.side, m.question[:50])
                continue
            committed += cost
            wanted.append(q)

        stale, unmatched = [], list(wanted)
        for o in self.broker.open_orders(m.condition_id):
            hit = next((q for q in unmatched if self._matches(o, q, tick)), None)
            if hit:
                unmatched.remove(hit)
            else:
                stale.append(o["id"])
        self.broker.cancel(stale)
        for q in unmatched:
            self._place(q, m)
        return committed

    # -- loop ---------------------------------------------------------------
    def cycle(self) -> None:
        notional_left, failed = self.cfg.max_open_notional, False
        for m in self.markets:
            try:
                notional_left -= self._quote_market(m, notional_left)
            except Exception:
                failed = True
                log.exception("cycle failed for %s", m.question[:50])
        self._errors = self._errors + 1 if failed else 0

    def _refresh_markets(self) -> None:
        fresh = select_markets(self.cfg)
        keep = {m.condition_id for m in fresh}
        for m in self.markets:
            if m.condition_id not in keep:
                log.info("dropping market: %s", m.question[:50])
                self.broker.cancel_market(m.condition_id)
        self.markets = fresh
        for m in fresh:
            log.info("making: %s (24h vol $%.0f, rewards=%s)", m.question[:60], m.volume_24h, bool(m.rewards_max_spread))

    def run(self) -> int:
        last_select = float("-inf")
        try:
            while not self.stop.is_set():
                if time.monotonic() - last_select > RESELECT_SECONDS:
                    self._refresh_markets()
                    last_select = time.monotonic()
                self.cycle()
                if self._errors >= self.cfg.max_errors:
                    log.error("%d consecutive failed cycles, shutting down", self._errors)
                    return 1
                self.stop.wait(self.cfg.refresh_seconds)
            return 0
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("cancelling all quotes")
        for m in self.markets:
            try:
                self.broker.cancel_market(m.condition_id)
            except Exception:
                log.exception("failed to cancel quotes for %s - CHECK OPEN ORDERS MANUALLY", m.question[:50])
