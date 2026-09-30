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
    take_slack: float = 0.06       # never take an exit more than this below fair
    min_p: float = 0.10
    max_p: float = 0.90
    loop_hz: float = 4.0


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


def plan_entry(kb, pb, fair: float, cfg: PairConfig):
    """Best pairing of two resting maker bids, or None. kb/pb are the two venues' YES books.
    Combo A: Up bid at Kalshi's best bid + Down bid at Polymarket's best Down bid (= YES ask). Combo B: the reverse."""
    if None in (kb.best_bid, kb.best_ask, pb.best_bid, pb.best_ask):
        return None
    options = {
        "Up@Kalshi+Down@Polymarket": (Leg(K, BUY_LONG, kb.best_bid, cfg.size), Leg(P, BUY_SHORT, pb.best_ask, cfg.size)),
        "Up@Polymarket+Down@Kalshi": (Leg(P, BUY_LONG, pb.best_bid, cfg.size), Leg(K, BUY_SHORT, kb.best_ask, cfg.size)),
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


class PairMaker:
    def __init__(self, cfg: PairConfig, us: UsAuth, kal: KalshiAuth, brti: BrtiFeed, sampler: SecondSampler):
        self.cfg, self.us, self.kal, self.brti, self.sampler = cfg, us, kal, brti, sampler
        self.kfeed = KalshiBookFeed(kal)
        self.pm = None
        self.pfeed: UsBookFeed | None = None
        self.brokers = {K: PaperBroker(maker_rebate=-0.0175, taker_fee=0.07), P: PaperBroker()}
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.first_fill: float | None = None
        self.total_pnl, self.windows, self.stats = 0.0, 0, {"pairs": 0, "legged": 0, "no_fill": 0}
        self._last_log = 0.0

    # ---- helpers ---------------------------------------------------------------------------------------------
    def _net(self) -> dict:
        return {v: b.position() for v, b in self.brokers.items()}

    def _on_book(self, venue: str):
        def cb(book):
            with self._lock:
                self.brokers[venue].on_book(book.best_bid, book.best_ask)
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
        self.first_fill = None
        log.info("=== window %s (ref %s) ===", nxt.slug[-16:], nxt.price_to_beat)
        return True

    def _settle(self, now: float) -> None:
        m = self.pm
        end_avg, ref = self.sampler.window_avg(m.window_end), m.price_to_beat
        if end_avg is None or ref is None:
            log.warning("cannot settle (missing BRTI data)")
            return
        up = end_avg >= ref
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
        fair = fair_up(spot=spot, k=ref, now=now, window_end=m.window_end, sigma=self.sampler.sigma(now), sampler=self.sampler)
        net = self._net()
        total = net[K] + net[P]
        t_in = now - m.window_start
        with self._lock:
            # 1) a lone leg: wait for the partner, then exit
            if abs(total) > 1e-9:
                if self.first_fill is None:
                    self.first_fill = now
                if now - self.first_fill >= c.leg_timeout or now > m.window_end - c.stop_before_end:
                    self._exit_lone_leg(net, fair, kb, pb)
                return
            self.first_fill = None
            # 2) flat or fully hedged: rest a pair if the window/price is suitable
            quoting = (c.start_delay <= t_in <= (m.window_end - m.window_start) - c.stop_before_end
                       and c.min_p <= fair <= c.max_p)
            plan = plan_entry(kb, pb, fair, c) if quoting and abs(net[K]) < 1e-9 and abs(net[P]) < 1e-9 else None
            self._sync_orders(plan)
            if now - self._last_log > 10:
                log.info("T-%4.0fs fair %.3f | K %s/%s  P %s/%s | %s", m.window_end - now, fair, _f(kb.best_bid), _f(kb.best_ask),
                         _f(pb.best_bid), _f(pb.best_ask), f"PAIR {plan[1]} margin {plan[0] * 100:.1f}c" if plan else "no pair (margin/zone)")
                self._last_log = now

    def _sync_orders(self, plan) -> None:
        want = {}
        if plan:
            _, _, up, dn = plan
            for leg in (up, dn):
                want[leg.venue] = leg
        for v, broker in self.brokers.items():
            have = broker.open_orders()
            leg = want.get(v)
            if leg and len(have) == 1 and have[0].intent == leg.intent and abs(have[0].price - leg.price) < 1e-6:
                continue
            for o in have:
                broker.cancel(o.id)
            if leg:
                broker.place(leg.intent, leg.price, leg.qty)

    def _exit_lone_leg(self, net, fair, kb, pb) -> None:
        """Cancel the waiting partner order and exit the filled leg by crossing the spread (IOC) if not absurdly bad."""
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
    maker = PairMaker(PairConfig(), us, kal, brti, sampler)
    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: maker.stop.set())
    maker.run()


if __name__ == "__main__":
    main()
