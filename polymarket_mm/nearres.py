"""Near-resolution bot for the BTC 60-minute Up/Down market on Polymarket US.

Near the end of a window the outcome is often all but decided (BRTI far from the reference price) yet the winning side
still trades a few cents under $1. Buy it and hold to settlement. Backtest (14 days, 187 windows, net of fees, model
certainty >= 0.99, >= 2c edge): 153 trades, 100% wins, +5.6c per contract (95% CI +4.4..+7.3c).

THE RISK is one reversal: a loser costs ~95c and erases ~17 wins. Protection, also backtested: sell into the bid the moment
the model's probability for our side falls to `stop` (0.85): worst trade -15c instead of -95c at a cost of ~0.3c per trade.
Settlement averages the final 60 BRTI prints, so a reversal develops over tens of seconds, which is what makes an exit possible.

One trade per window, 1 contract, taker orders (IOC). Dry-run by default; --live trades.

  python -m polymarket_mm.nearres [--live]
"""
import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from .brti import BrtiFeed, KalshiAuth
from .hourly import HourlyMarket, find_current
from .model import SecondSampler, fair_up
from .us_broker import BUY_LONG, BUY_SHORT, SELL_LONG, SELL_SHORT, LiveBroker, UsRest
from .us_feed import UsAuth, UsBookFeed
from .us_private_feed import UsPrivateFeed

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NRConfig:
    size: float = 1.0
    theta: float = 0.99             # enter only when the model is at least this sure
    margin: float = 0.02            # ... and the market's ask is at least this far below the model's probability
    max_ask: float = 0.97           # never pay more than this
    stop: float = 0.85              # SELL if the model's probability for our side falls to this
    min_tau: float = 30.0           # no entry with less than this many seconds left
    max_tau: float = 900.0          # only in the last 15 minutes
    max_entry_tries: int = 6        # IOC attempts per window
    max_exit_tries: int = 12
    exit_slip_ticks: int = 2        # an exit's limit is this many ticks through the bid (we WANT out)
    max_session_loss: float = 0.60  # dollars: stop entering once the session is down this much
    loop_hz: float = 4.0


def entry_signal(fair: float, up_ask: float | None, dn_ask: float | None, tau: float, cfg: NRConfig):
    """('U'|'D', price) if we should buy the winning side now, else None. Prices are the side's own price."""
    if not (cfg.min_tau <= tau <= cfg.max_tau):
        return None
    if fair >= cfg.theta and up_ask is not None and up_ask <= cfg.max_ask and fair - up_ask >= cfg.margin:
        return "U", up_ask
    if fair <= 1 - cfg.theta and dn_ask is not None and dn_ask <= cfg.max_ask and (1 - fair) - dn_ask >= cfg.margin:
        return "D", dn_ask
    return None


def exit_signal(side: str, fair: float, cfg: NRConfig) -> bool:
    """True when the model's probability for the side we hold has fallen to the stop level."""
    return (fair if side == "U" else 1 - fair) <= cfg.stop


class NearResolutionBot:
    def __init__(self, cfg: NRConfig, us: UsAuth, brti: BrtiFeed, sampler: SecondSampler, rest: UsRest | None):
        self.cfg, self.us, self.brti, self.sampler, self.rest = cfg, us, brti, sampler, rest
        self.live = rest is not None
        self.mkt: HourlyMarket | None = None
        self.book_feed: UsBookFeed | None = None
        self.priv: UsPrivateFeed | None = None
        self.broker: LiveBroker | None = None
        self.stop = threading.Event()
        self.state = "idle"               # idle -> held -> done   (per window)
        self.side: str | None = None
        self.entry_px = 0.0
        self.entry_tries = self.exit_tries = 0
        self.session_pnl = 0.0
        self.windows = self.trades = self.stops = 0
        self.killed = False
        self._last_log = 0.0

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _roll(self, now: float) -> bool:
        nxt = find_current(self.us, now)
        if nxt is None:
            return False
        if self.book_feed:
            self.book_feed.stop()
        if self.priv:
            self.priv.stop()
        self.mkt = nxt
        self.book_feed = UsBookFeed(self.us, [nxt.slug])
        self.book_feed.start()
        if self.live:
            self.broker = LiveBroker(self.rest, nxt.slug)                   # no resting orders, so no expiry needed
            self.priv = UsPrivateFeed(self.us, [nxt.slug])
            self.priv.on_event = lambda _j: self.broker.poke()
            self.priv.start()
            self.broker.cancel_all()
        self.state, self.side, self.entry_tries, self.exit_tries = "idle", None, 0, 0
        log.info("=== window %s (ref %s) ===", nxt.slug[-16:], nxt.price_to_beat)
        return True

    def _settle_log(self, now: float) -> None:
        m = self.mkt
        self.windows += 1
        if self.state != "held" or self.side is None:
            return
        end_avg = self.sampler.window_avg(m.window_end)
        ref = m.price_to_beat
        if end_avg is None or ref is None:
            log.warning("held to settlement but cannot determine the outcome")
            return
        up = end_avg >= ref
        won = up if self.side == "U" else not up
        pnl = (1.0 if won else 0.0) - self.entry_px
        self.session_pnl += pnl * self.cfg.size
        log.info("SETTLED %s: %s %s | entry %.2f -> %+.2f | session %+.2f", m.slug[-16:], "UP" if up else "DOWN",
                 "WON" if won else "LOST", self.entry_px, pnl, self.session_pnl)

    # ---- one iteration -----------------------------------------------------------------------------------------
    def _step(self) -> None:
        now, c = time.time(), self.cfg
        if self.mkt is None or now >= self.mkt.window_end + 3:
            if self.mkt:
                self._settle_log(now)
            if not self._roll(now):
                time.sleep(3)
                return
        m = self.mkt
        bk = self.book_feed.book(m.slug)
        spot, ref = self.sampler.last(), m.price_to_beat
        if not (bk and bk.best_bid is not None and bk.best_ask is not None and spot and ref and self.brti.latest()):
            return
        tau = m.window_end - now
        fair = fair_up(spot=spot, k=ref, now=now, window_end=m.window_end, sigma=self.sampler.sigma(now), sampler=self.sampler)
        up_ask, dn_ask = bk.best_ask, 1 - bk.best_bid

        if self.state == "idle" and not self.killed and self.entry_tries < c.max_entry_tries:
            sig = entry_signal(fair, up_ask, dn_ask, tau, c)
            if sig:
                side, px = sig
                intent, limit = (BUY_LONG, bk.best_ask) if side == "U" else (BUY_SHORT, bk.best_bid)
                self.entry_tries += 1
                log.warning("ENTRY: model %.3f sure of %s, market asks %.3f (%.1fc below) T-%.0fs -> %s", max(fair, 1 - fair),
                            "UP" if side == "U" else "DOWN", px, ((fair if side == 'U' else 1 - fair) - px) * 100, tau,
                            "buying" if self.live else "DRY-RUN: would buy")
                ok = (not self.live) or self.broker.take(intent, limit, c.size)
                if ok:
                    self.state, self.side, self.entry_px = "held", side, px
                    self.trades += 1
        elif self.state == "held":
            if self.live and abs(self.broker.position()) < 0.01:          # nothing left to manage (should not happen)
                log.warning("position is flat but we thought we were holding: standing down for this window")
                self.state = "done"
                return
            if exit_signal(self.side, fair, c) and self.exit_tries < c.max_exit_tries:
                intent, limit = ((SELL_LONG, round(bk.best_bid - 0.01 * c.exit_slip_ticks, 2)) if self.side == "U"
                                 else (SELL_SHORT, round(bk.best_ask + 0.01 * c.exit_slip_ticks, 2)))
                self.exit_tries += 1
                mine = fair if self.side == "U" else 1 - fair
                log.warning("STOP: model's probability for %s fell to %.3f (<= %.2f) -> selling (try %d)",
                            "UP" if self.side == "U" else "DOWN", mine, c.stop, self.exit_tries)
                ok = (not self.live) or self.broker.take(intent, limit, c.size)
                if ok:
                    bid = bk.best_bid if self.side == "U" else 1 - bk.best_ask
                    pnl = (bid - self.entry_px) * c.size
                    self.session_pnl += pnl
                    self.stops += 1
                    self.state = "done"
                    log.warning("STOPPED OUT: ~%+.2f | session %+.2f", pnl, self.session_pnl)
        if self.session_pnl <= -c.max_session_loss and not self.killed:
            self.killed = True
            log.critical("SESSION LOSS LIMIT hit (%.2f): no more entries this session", self.session_pnl)
        if now - self._last_log > 10:
            log.info("T-%4.0fs BRTI %.2f ref %.2f | model UP %.3f | asks: Up %.2f Down %.2f | %s%s", tau, spot, ref, fair, up_ask, dn_ask,
                     self.state, f" ({'UP' if self.side == 'U' else 'DOWN'} @ {self.entry_px:.2f})" if self.state == "held" else "")
            self._last_log = now

    def run(self) -> None:
        errors = 0
        try:
            while not self.stop.is_set():
                try:
                    self._step()
                    errors = 0
                except Exception as e:
                    errors += 1
                    log.warning("loop error %d: %s: %s", errors, type(e).__name__, str(e)[:120])
                    if errors >= 40:
                        log.critical("too many errors, giving up")
                        break
                    time.sleep(min(2.0, 0.25 * errors))
                    continue
                time.sleep(1 / self.cfg.loop_hz)
        finally:
            if self.book_feed:
                self.book_feed.stop()
            if self.priv:
                self.priv.stop()
            self.brti.stop()
            log.info("FINAL: windows %d | trades %d | stops %d | session %+.2f", self.windows, self.trades, self.stops, self.session_pnl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place REAL orders (default: dry-run, logs what it would do)")
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
    if args.live:
        log.warning("LIVE MODE: real orders, 1 contract, theta 0.99, stop 0.85")
    bot = NearResolutionBot(NRConfig(), us, brti, sampler, UsRest(us) if args.live else None)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: bot.stop.set())
    bot.run()


if __name__ == "__main__":
    main()
