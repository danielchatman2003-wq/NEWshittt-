import pytest

from polymarket_mm.nearres import NRConfig, NearResolutionBot, entry_signal, exit_signal
from polymarket_mm.us_feed import UsBook

C = NRConfig()


def test_entry_needs_certainty_edge_and_the_right_time():
    assert entry_signal(0.995, 0.96, 0.05, 300, C) == ("U", 0.96)              # sure + 3.5c edge -> buy Up
    assert entry_signal(0.005, 0.05, 0.96, 300, C) == ("D", 0.96)              # the mirror: buy Down
    assert entry_signal(0.97, 0.94, 0.07, 300, C) is None                      # not sure enough (0.97 < 0.99)
    assert entry_signal(0.995, 0.985, 0.02, 300, C) is None                    # edge only 1c: market already prices it
    assert entry_signal(0.995, 0.98, 0.02, 300, C) is None                     # above max_ask 0.97
    assert entry_signal(0.995, 0.96, 0.05, 10, C) is None                      # too late to react
    assert entry_signal(0.995, 0.96, 0.05, 1500, C) is None                    # too early
    assert entry_signal(0.995, None, None, 300, C) is None                     # no book


def test_exit_fires_when_the_models_confidence_in_our_side_falls():
    assert not exit_signal("U", 0.98, C) and not exit_signal("U", 0.86, C)
    assert exit_signal("U", 0.85, C) and exit_signal("U", 0.40, C)
    assert not exit_signal("D", 0.02, C) and exit_signal("D", 0.15, C)         # Down: our probability is 1 - fair


class FakeBroker:
    def __init__(self):
        self.takes, self.pos = [], 0.0

    def take(self, intent, price, qty):
        self.takes.append((intent, price, qty))
        self.pos += qty if intent.endswith(("BUY_LONG", "SELL_SHORT")) else -qty
        return True

    def position(self):
        return self.pos


def make_bot(fair_inputs, book=(0.94, 0.95), live=True):
    import threading
    from polymarket_mm.hourly import HourlyMarket
    b = object.__new__(NearResolutionBot)
    b.cfg, b.live, b.state, b.side, b.entry_px = C, live, "idle", None, 0.0
    b.entry_tries = b.exit_tries = b.trades = b.stops = b.windows = 0
    b.session_pnl, b.killed, b._last_log = 0.0, False, 0.0
    b.stop = threading.Event()
    b.broker = FakeBroker()
    b.mkt = HourlyMarket("s", 0.0, 4000.0, 84000.0, 0.01, 0.01, 0.0695, True)
    b.book_feed = type("F", (), {"book": lambda self, s: UsBook("s", bids=[(book[0], 50)], asks=[(book[1], 50)])})()
    b.brti = type("B", (), {"latest": lambda self: object()})()
    b.sampler = type("S", (), {"last": lambda self: fair_inputs["spot"], "sigma": lambda self, t: 6.0, "px": {}, "window_avg": lambda self, t: None})()
    return b


def run_step(b, now):
    import time as _t
    orig = _t.time
    _t.time = lambda: now
    try:
        b._step()
    finally:
        _t.time = orig


def test_bot_buys_up_when_far_above_the_reference_and_holds():
    b = make_bot({"spot": 84600.0}, book=(0.95, 0.96))               # $600 above ref, 300s left: model ~1.0; Up ask 0.96
    run_step(b, 3700.0)
    assert b.broker.takes == [("ORDER_INTENT_BUY_LONG", 0.96, 1.0)] and b.state == "held" and b.side == "U"
    run_step(b, 3701.0)                                              # still safe: no further orders
    assert len(b.broker.takes) == 1


def test_bot_buys_down_when_far_below_the_reference():
    b = make_bot({"spot": 83400.0}, book=(0.03, 0.04))               # Up bid 0.03 -> Down ask 0.97? use a cheaper Down: bid 0.05
    b.book_feed = type("F", (), {"book": lambda self, s: UsBook("s", bids=[(0.04, 50)], asks=[(0.05, 50)])})()   # Down ask = 0.96
    run_step(b, 3700.0)
    assert b.broker.takes == [("ORDER_INTENT_BUY_SHORT", 0.04, 1.0)] and b.side == "D"


def test_stop_sells_into_the_bid_when_the_model_turns():
    b = make_bot({"spot": 84600.0}, book=(0.95, 0.96))
    run_step(b, 3700.0)                                              # bought Up at 0.96
    b.sampler = type("S", (), {"last": lambda self: 84000.0, "sigma": lambda self, t: 6.0, "px": {}, "window_avg": lambda self, t: None})()
    b.book_feed = type("F", (), {"book": lambda self, s: UsBook("s", bids=[(0.55, 50)], asks=[(0.57, 50)])})()
    run_step(b, 3720.0)                                              # BRTI collapsed to the reference: model ~0.5 -> stop
    assert b.broker.takes[-1][0] == "ORDER_INTENT_SELL_LONG" and b.state == "done" and b.stops == 1
    assert b.broker.takes[-1][1] == pytest.approx(0.55 - 0.02)       # limit goes THROUGH the bid: we want out
    assert b.session_pnl == pytest.approx(0.55 - 0.96)
    n = len(b.broker.takes)
    run_step(b, 3721.0)
    assert len(b.broker.takes) == n                                  # one trade per window: no re-entry


def test_no_entry_when_the_market_is_not_cheap():
    b = make_bot({"spot": 84600.0}, book=(0.985, 0.99))              # already priced at ~0.99: no edge
    run_step(b, 3700.0)
    assert b.broker.takes == [] and b.state == "idle"


def test_dry_run_places_nothing():
    b = make_bot({"spot": 84600.0}, book=(0.95, 0.96), live=False)
    run_step(b, 3700.0)
    assert b.broker.takes == [] and b.state == "held"                # it 'would' have bought; no order was sent


def test_session_loss_limit_blocks_new_entries():
    b = make_bot({"spot": 84600.0}, book=(0.95, 0.96))
    b.session_pnl = -0.7
    run_step(b, 3700.0)                                              # trips the kill switch this step (after any entry decision)
    assert b.killed
    b.state, b.entry_tries = "idle", 0
    before = len(b.broker.takes)
    run_step(b, 3701.0)
    assert len(b.broker.takes) == before
