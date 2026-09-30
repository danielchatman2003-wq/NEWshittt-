"""Cross-venue MAKER PAIR on the BTC 15-minute Up/Down contract (Kalshi + Polymarket US list the identical contract).

Idea: rest a maker bid for UP on one venue and a maker bid for DOWN on the other such that the two prices sum to
less than $1.  If BOTH fill, the pair pays exactly $1 -> locked profit (= the margin, plus/minus tiny maker fees).
Choose the pairing (Up@Kalshi+Down@Polymarket, or the reverse) with the bigger margin.
If only ONE leg fills we hold a directional leg: keep the other leg resting for `leg_timeout` seconds, then
cancel it and exit the filled leg by crossing the spread (the legging risk that has to be measured).

  python -m polymarket_mm.pairmaker15            paper (real books, simulated fills, settled from real BRTI)
"""
import argparse
import logging
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from .brti import BrtiFeed, KalshiAuth
from .hourly import find_current as find_pm
from .kalshi_feed import KalshiBookFeed
from .model import SecondSampler, fair_up
from .us_broker import BUY_LONG, BUY_SHORT, SELL_LONG, SELL_SHORT, PaperBroker
from .us_feed import UsAuth, UsBookFeed

log = logging.getLogger(__name__)
K, P = "Kalshi", "Polymarket"


@dataclass(frozen=True)
class PairConfig:
    size: float = 1.0
    min_margin: float = 0.01       # both legs rest only if the two bids sum to <= 1 - min_margin
    tol: float = 0.03              # a leg may be at most this far above the model's fair value for its side
    start_delay: float = 120.0     # seconds after the window opens before quoting
    stop_before_end: float = 150.0
    leg_timeout: float = 25.0      # how long a lone filled leg waits for its partner before we exit it
    hedge_max_cost: float = 1.03   # when one leg fills, BUY THE OTHER SIDE AT ONCE (taker) if the pair still costs <= this
    take_slack: float = 0.06       # never take an exit more than this below fair
    min_p: float = 0.10
    max_p: float = 0.90
    loop_hz: float = 4.0
    # --- money limits (hard) ---
    max_capital: float = 2.0       # most money committed to ONE window's pair (both venues, both legs, incl. the loaded size)
    max_loss: float = 1.0          # session kill switch: stop quoting and cancel everything once down this much
    # --- directional loading ---
    lean_edge: float = 0.02        # model fair vs the books' mid must differ by this much to pick a direction
    lean_extra: float = 1.0        # extra size (in multiples of `size`) loaded onto the favoured side's leg


@dataclass(frozen=True)
class Leg:
    venue: str       # K or P
    intent: str      # BUY_LONG (bid Up) or BUY_SHORT (bid Down, expressed at the YES price)
    price: float     # YES price
    qty: float

    @property
    def side_price(self) -> float:
        """Price OF THE SIDE being bought (Up price for BUY_LONG, Down price for BUY_SHORT)."""
        return self.price if self.intent == BUY_LONG else 1 - self.price


def _f(x) -> str:
    return "  -- " if x is None else f"{x:.3f}"


def direction_edge(kb, pb, fair: float) -> float | None:
    """Model minus market for Up: fair - average of the two books' mids. >0: the model thinks Up is underpriced."""
    mids = [b.mid for b in (kb, pb) if b is not None and b.mid is not None]
    return fair - sum(mids) / len(mids) if mids else None


def leg_sizes(edge: float | None, cfg: PairConfig) -> tuple[float, float]:
    """(up_qty, down_qty). Directional loading: extra size on the side the model favours, base size on the other."""
    if edge is None:
        return cfg.size, cfg.size
    extra = cfg.size * cfg.lean_extra
    if edge >= cfg.lean_edge:
        return cfg.size + extra, cfg.size
    if edge <= -cfg.lean_edge:
        return cfg.size, cfg.size + extra
    return cfg.size, cfg.size


def pair_cost(up: "Leg", dn: "Leg") -> float:
    """Money committed if both legs fill (each leg pays its own side's price)."""
    return up.side_price * up.qty + dn.side_price * dn.qty


def phase(net: dict) -> str:
    """flat: nothing held | lone: exactly one venue holds a leg | paired: both venues hold one (the normal end state)."""
    held = [v for v, q in net.items() if abs(q) > 1e-9]
    return "flat" if not held else "lone" if len(held) == 1 else "paired"


def plan_entry(kb, pb, fair: float, cfg: PairConfig, up_qty: float | None = None, dn_qty: float | None = None):
    """Best pairing of two resting maker bids, or None. kb/pb are the two venues' YES books.
    Combo A: Up bid at Kalshi's best bid + Down bid at Polymarket's best Down bid (= YES ask). Combo B: the reverse."""
    if None in (kb.best_bid, kb.best_ask, pb.best_bid, pb.best_ask):
        return None
    uq, dq = (up_qty or cfg.size), (dn_qty or cfg.size)
    options = {
        "Up@Kalshi+Down@Polymarket": (Leg(K, BUY_LONG, kb.best_bid, uq), Leg(P, BUY_SHORT, pb.best_ask, dq)),
        "Up@Polymarket+Down@Kalshi": (Leg(P, BUY_LONG, pb.best_bid, uq), Leg(K, BUY_SHORT, kb.best_ask, dq)),
    }
    best = None
    for name, (up, dn) in options.items():
        total = up.side_price + dn.side_price
        margin = 1 - total
        if margin < cfg.min_margin - 1e-9:
            continue
        if up.side_price > fair + cfg.tol or dn.side_price > (1 - fair) + cfg.tol:   # the model must not hate either price
            continue
        if best is None or margin > best[0]:
            best = (margin, name, up, dn)
    return best


def choose_plan(kb, pb, fair: float, cfg: PairConfig):
    """The pair to rest right now, sized by direction and trimmed to the money cap; None if nothing fits."""
    uq, dq = leg_sizes(direction_edge(kb, pb, fair), cfg)
    for up_qty, dn_qty in ((uq, dq), (cfg.size, cfg.size)):             # try the loaded size first, then the base size
        plan = plan_entry(kb, pb, fair, cfg, up_qty, dn_qty)
        if plan and pair_cost(plan[2], plan[3]) <= cfg.max_capital + 1e-9:
            return plan
    return None


class PairMaker:
    def __init__(self, cfg: PairConfig, us: UsAuth, kal: KalshiAuth, brti: BrtiFeed, sampler: SecondSampler,
                 prest=None, krest=None):
        self.cfg, self.us, self.kal, self.brti, self.sampler = cfg, us, kal, brti, sampler
        self.prest, self.krest, self._k_ticker = prest, krest, None
        self.kfeed = KalshiBookFeed(kal)
        self.pm = None
        self.pfeed: UsBookFeed | None = None
        self.brokers = {} if (prest is not None and krest is not None) else {K: PaperBroker(maker_rebate=-0.0175, taker_fee=0.07), P: PaperBroker()}
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.live = prest is not None and krest is not None
        self.killed = False
        self._last_cash = 0.0
        self._equity0: float | None = None
        self.first_fill: float | None = None
        self._plan_legs: dict = {}      # venue -> the Leg most recently rested there
        self._completed = False         # already tried to complete the pair for this lone leg
        self.window_done = False   # one attempt per window: after a lone-leg exit we stand down until the next window
        self.total_pnl, self.windows, self.stats = 0.0, 0, {"pairs": 0, "legged": 0, "no_fill": 0}
        self._last_log = 0.0

    # ---- helpers ---------------------------------------------------------------------------------------------
    def _cash(self) -> float:
        pm = float(self.prest.call("GET", "/v1/account/balances")["balances"][0]["currentBalance"])
        ka = float(self.krest.call("GET", "/portfolio/balance")["balance"]) / 100.0
        return pm + ka

    def _ensure_brokers(self, km) -> bool:
        """Live only: build this window's brokers once Kalshi's feed is on the same market as Polymarket's."""
        if not self.live or (self.brokers and self._k_ticker == km.ticker):
            return True
        if abs(km.close_ts - self.pm.window_end) > 3:
            return False
        end = self.pm.window_end - self.cfg.stop_before_end
        from .kalshi_broker import KalshiLiveBroker
        from .us_broker import LiveBroker
        self.brokers = {K: KalshiLiveBroker(self.krest, km.ticker, expire_at=end), P: LiveBroker(self.prest, self.pm.slug, expire_at=end)}
        self._k_ticker = km.ticker
        for b in self.brokers.values():
            b.cancel_all()
        log.warning("LIVE brokers ready: %s + %s | positions K %+g P %+g | cash $%.2f", km.ticker, self.pm.slug[-16:],
                    self.brokers[K].position(), self.brokers[P].position(), self._cash())
        return True

    def _net(self) -> dict:
        return {v: b.position() for v, b in self.brokers.items()}

    def _on_book(self, venue: str):
        def cb(book):
            if self.live:
                return                      # real fills come from the exchanges; book-driven fill simulation is paper-only
            with self._lock:
                b = self.brokers.get(venue)
                if b is not None:
                    b.on_book(book.best_bid, book.best_ask)
        return cb

    def _roll(self, now: float) -> bool:
        nxt = find_pm(self.us, now, horizon="15m")
        if nxt is None:
            return False
        if self.pfeed:
            self.pfeed.stop()
        self.pm, self.pfeed = nxt, UsBookFeed(self.us, [nxt.slug])
        self.pfeed.on_update = self._on_book(P)
        self.pfeed.start()
        self.kfeed.on_update = self._on_book(K)
        for b in self.brokers.values():
            b.cancel_all()
        if self.live:
            self.brokers, self._k_ticker = {}, None      # fresh live brokers per window (per-market tickers/slugs)
        self.first_fill = None
        self.window_done = False
        self._completed = False
        self._plan_legs = {}
        log.info("=== window %s (ref %s) ===", nxt.slug[-16:], nxt.price_to_beat)
        return True

    def _settle(self, now: float) -> None:
        m = self.pm
        end_avg, ref = self.sampler.window_avg(m.window_end), m.price_to_beat
        if end_avg is None or ref is None:
            log.warning("cannot settle (missing BRTI data)")
            return
        up = end_avg >= ref
        if self.live:
            cash = self._cash()
            self.windows += 1
            log.info("WINDOW END %s -> %s | cash $%.2f (session start $%.2f => %+.2f)", m.slug[-16:], "UP" if up else "DOWN",
                     cash, self._equity0 or 0.0, cash - (self._equity0 or cash))
            return
        n = {v: b.pos for v, b in self.brokers.items()}
        fills = sum(len(b.fills) for b in self.brokers.values())
        pnl = sum(b.settle(up) for b in self.brokers.values())
        self.total_pnl += pnl
        self.windows += 1
        kind = "hedged pair" if fills >= 2 and abs(sum(n.values())) < 1e-9 else "UNHEDGED" if fills else "no fills"
        self.stats["pairs" if kind == "hedged pair" else "legged" if kind == "UNHEDGED" else "no_fill"] += 1
        log.info("SETTLED %s -> %s | fills=%d (%s) P&L $%+.3f | session $%+.3f over %d windows  [%s]", m.slug[-16:],
                 "UP" if up else "DOWN", fills, kind, pnl, self.total_pnl, self.windows, self.stats)

    # ---- the loop --------------------------------------------------------------------------------------------
    def _step(self) -> None:
        now, c = time.time(), self.cfg
        if self.pm is None or now >= self.pm.window_end + 3:
            if self.pm:
                with self._lock:
                    for b in self.brokers.values():
                        b.cancel_all()
                self._settle(now)
            if not self._roll(now):
                time.sleep(3)
                return
        m = self.pm
        kb, pb, km = self.kfeed.book(), self.pfeed.book(m.slug), self.kfeed.market
        spot, ref = self.sampler.last(), m.price_to_beat
        if not (kb and pb and km and spot and ref and self.brti.latest()):
            return
        if abs(km.reference - ref) > 0.005:
            return                                     # the two venues must be the same contract: refuse if references differ
        if not self._ensure_brokers(km):
            return
        fair = fair_up(spot=spot, k=ref, now=now, window_end=m.window_end, sigma=self.sampler.sigma(now), sampler=self.sampler)
        net = self._net()
        total = net[K] + net[P]
        t_in = now - m.window_start
        edge = direction_edge(kb, pb, fair)
        with self._lock:
            ph = phase(net)
            if ph == "paired":                                   # both legs filled: done for this window, hold to settlement
                for b in self.brokers.values():
                    if b.open_orders():
                        b.cancel_all()
                if now - self._last_log > 15:
                    log.info("T-%4.0fs PAIRED %s — holding to settlement (fair %.3f)", m.window_end - now, net, fair)
                    self._last_log = now
                return
            if ph == "lone":                                     # 1) one leg filled: hedge it at once, else wait, then exit
                if self._try_complete(net, kb, pb):
                    return
                if self.first_fill is None:
                    self.first_fill = now
                if now - self.first_fill >= c.leg_timeout or now > m.window_end - c.stop_before_end:
                    self._exit_lone_leg(net, fair, kb, pb)
                return
            self.first_fill = None
            if self.window_done or self.killed or self._check_loss(now):
                self._sync_orders(None)
                return
            # 2) flat: rest a pair if the window/price is suitable (one pair per window)
            quoting = (c.start_delay <= t_in <= (m.window_end - m.window_start) - c.stop_before_end
                       and c.min_p <= fair <= c.max_p)
            plan = choose_plan(kb, pb, fair, c) if quoting else None
            self._sync_orders(plan)
            if now - self._last_log > 10:
                side = "--" if edge is None else ("UP" if edge >= c.lean_edge else "DOWN" if edge <= -c.lean_edge else "neutral")
                log.info("T-%4.0fs fair %.3f (model-vs-books %s%.1fc => %s) | K %s/%s  P %s/%s | %s", m.window_end - now, fair,
                         "" if edge is None else "%+.1f" % (edge * 100), "", side, _f(kb.best_bid), _f(kb.best_ask),
                         _f(pb.best_bid), _f(pb.best_ask),
                         f"PAIR {plan[1]} margin {plan[0] * 100:.1f}c sizes {plan[2].qty:g}/{plan[3].qty:g} cost ${pair_cost(plan[2], plan[3]):.2f}" if plan else "no pair (margin/zone)")
                self._last_log = now

    def _check_loss(self, now: float) -> bool:
        """Session kill switch. Paper: settled P&L. Live: cash now vs cash at start (only read while flat with nothing resting)."""
        if self.cfg.max_loss <= 0:
            return False
        if self.live:
            if any(b.open_orders() for b in self.brokers.values()) or now - self._last_cash < 30 or now - self.pm.window_start < 100:
                return False                     # cash is only a clean measure when flat, nothing resting, and last window paid out
            self._last_cash = now
            loss = (self._equity0 - self._cash()) if self._equity0 is not None else 0.0
        else:
            loss = -self.total_pnl
        if loss >= self.cfg.max_loss:
            self.killed = True
            log.critical("KILL SWITCH: session loss $%.2f >= limit $%.2f. No more orders this session.", loss, self.cfg.max_loss)
            for b in self.brokers.values():
                try:
                    b.cancel_all()
                except Exception:
                    log.exception("cancel during kill switch failed")
            return True
        return False

    def _sync_orders(self, plan) -> None:
        want = {}
        if plan:
            _, _, up, dn = plan
            for leg in (up, dn):
                want[leg.venue] = leg
                self._plan_legs[leg.venue] = leg
        for v, broker in self.brokers.items():
            have = broker.open_orders()
            leg = want.get(v)
            if leg and len(have) == 1 and have[0].intent == leg.intent and abs(have[0].price - leg.price) < 1e-6:
                continue
            for o in have:
                broker.cancel(o.id)
            if leg:
                broker.place(leg.intent, leg.price, leg.qty)

    def _try_complete(self, net, kb, pb) -> bool:
        """One leg filled: buy the OTHER side right now (IOC taker on the other venue) if the pair still costs
        <= hedge_max_cost. Both legs then pay exactly $1 whatever happens, so the result is locked (near break-even)
        instead of an open directional leg. Returns True if the hedge order was sent."""
        if self._completed:
            return False
        held = next((v for v, q in net.items() if abs(q) > 1e-9), None)
        if held is None:
            return False
        other = P if held == K else K
        entry, partner = self._plan_legs.get(held), self._plan_legs.get(other)
        book = pb if other == P else kb
        if not entry or not partner:
            return False
        tick = 0.01
        if net[held] > 0:                              # holding Up -> buy Down on the other venue (= sell YES at its bid)
            if book.best_bid is None:
                return False
            combined, intent, px = entry.side_price + (1 - book.best_bid), BUY_SHORT, max(tick, book.best_bid - tick)
        else:                                          # holding Down -> buy Up on the other venue (= buy YES at its ask)
            if book.best_ask is None:
                return False
            combined, intent, px = entry.side_price + book.best_ask, BUY_LONG, min(1 - tick, book.best_ask + tick)
        if combined > self.cfg.hedge_max_cost:
            return False                               # the market already ran away: fall through to wait/unwind
        self._completed = True
        self.brokers[other].cancel_all()               # drop the resting partner bid so it cannot double-fill
        ok = self.brokers[other].take(intent, px, partner.qty)
        log.warning("COMPLETE PAIR: holding %s on %s (entry %.3f) -> buying the other side on %s (limit %.2f) | combined ~%.3f %s",
                    "Up" if net[held] > 0 else "Down", held, entry.side_price, other, px, combined, "sent" if ok else "FAILED")
        return ok

    def _exit_lone_leg(self, net, fair, kb, pb) -> None:
        """Cancel the waiting partner order and exit the filled leg by crossing the spread (IOC) if not absurdly bad."""
        self.window_done = True        # one legging attempt per window: don't immediately try again into the same move
        for b in self.brokers.values():
            b.cancel_all()
        for v, book in ((K, kb), (P, pb)):
            q = net[v]
            if abs(q) < 1e-9:
                continue
            if q > 0:                                   # long Up here: sell at the best bid
                px = book.best_bid
                ok = px is not None and px >= fair - self.cfg.take_slack
                if ok:
                    self.brokers[v].take(SELL_LONG, px, q)
            else:                                       # long Down here: sell Down = buy Up at the YES ask
                px = book.best_ask
                ok = px is not None and px <= fair + self.cfg.take_slack
                if ok:
                    self.brokers[v].take(SELL_SHORT, px, -q)
            log.warning("LONE LEG on %s (%+g): exit %s", v, q, "taken" if ok else "skipped (model says hold)")

    def run(self) -> None:
        self.kfeed.start()
        if self.live:
            self._equity0 = self._cash()
            log.warning("LIVE MODE: session start cash $%.2f (limits: capital/window $%.2f, loss stop $%.2f)", self._equity0,
                        self.cfg.max_capital, self.cfg.max_loss)
        try:
            while not self.stop.is_set():
                try:
                    self._step()
                except Exception as e:
                    log.warning("step error: %s: %s", type(e).__name__, str(e)[:120])
                    time.sleep(1)
                time.sleep(1 / self.cfg.loop_hz)
        finally:
            for b in self.brokers.values():
                b.cancel_all()
            self.kfeed.stop()
            if self.pfeed:
                self.pfeed.stop()
            self.brti.stop()
            log.info("FINAL: session P&L $%+.3f over %d windows %s", self.total_pnl, self.windows, self.stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--live", action="store_true", help="place REAL orders on both venues")
    ap.add_argument("--max-capital", type=float, default=PairConfig.max_capital, help="most money committed to one window's pair")
    ap.add_argument("--max-loss", type=float, default=PairConfig.max_loss, help="session kill switch: stop after losing this much")
    args = ap.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    for n in ("websockets", "urllib3"):
        logging.getLogger(n).setLevel(logging.WARNING)
    us, kal = UsAuth.from_env(), KalshiAuth.from_env()
    from .hourly_mm import seed_sampler
    sampler = SecondSampler()
    log.info("seeded %d s of BRTI history", seed_sampler(sampler, kal))
    brti = BrtiFeed(kal)
    brti.on_tick = lambda t: t.spot and sampler.add(t.local_ts, t.spot)
    brti.start()
    cfg = PairConfig(max_capital=args.max_capital, max_loss=args.max_loss)
    prest = krest = None
    if args.live:
        from .kalshi_broker import KalshiRest
        from .us_broker import UsRest
        prest, krest = UsRest(us), KalshiRest(kal)
    maker = PairMaker(cfg, us, kal, brti, sampler, prest, krest)
    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: maker.stop.set())
    maker.run()


if __name__ == "__main__":
    main()
