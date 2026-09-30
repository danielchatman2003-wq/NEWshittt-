import time

from polymarket_mm.hourly_mm import HourlyConfig, yes_quotes
from polymarket_mm.us_broker import (BUY_LONG, BUY_SHORT, SELL_LONG, SELL_SHORT, PaperBroker, RateLimiter,
                                     order_body)

CFG = HourlyConfig(size=10, max_pos=50, skew=0.03)  # explicit sizes: tests the logic, not the live defaults


def q(pos=0.0, fair=0.50, half=0.02, cfg=CFG):
    return {x.intent: x for x in yes_quotes(fair=fair, half=half, tick=0.01, pos=pos, cfg=cfg)}


def test_flat_quotes_buy_up_and_buy_down_around_fair():
    r = q()
    assert set(r) == {BUY_LONG, BUY_SHORT}
    assert r[BUY_LONG].price == 0.48 and r[BUY_SHORT].price == 0.52 and r[BUY_LONG].qty == 10


def test_long_inventory_sells_and_skews_lower():
    r = q(pos=30)
    assert SELL_LONG in r and BUY_SHORT not in r and r[SELL_LONG].qty == 10
    assert r[BUY_LONG].price < q()[BUY_LONG].price and r[BUY_LONG].qty == 10


def test_short_inventory_unwinds_via_sell_short():
    r = q(pos=-4)
    assert r[SELL_SHORT].qty == 4 and BUY_LONG not in r and BUY_SHORT in r


def test_position_cap_stops_adding():
    r = q(pos=50)
    assert BUY_LONG not in r and SELL_LONG in r


def test_never_quotes_a_crossed_or_out_of_range_market():
    assert yes_quotes(fair=0.995, half=0.02, tick=0.01, pos=0, cfg=CFG) == []
    assert yes_quotes(fair=0.004, half=0.02, tick=0.01, pos=0, cfg=CFG) == []


def test_bid_below_ask_always():
    for fair in (0.1, 0.33, 0.5, 0.77, 0.9):
        r = q(fair=fair)
        assert r[BUY_LONG].price < r[BUY_SHORT].price


def test_order_body_is_maker_only_yes_priced():
    b = order_body("slug", BUY_SHORT, 0.52, 3)
    assert b["participateDontInitiate"] is True and b["price"]["value"] == "0.52"
    assert b["intent"] == BUY_SHORT and b["tif"] == "TIME_IN_FORCE_GOOD_TILL_CANCEL"


def test_paper_bid_fills_only_when_market_trades_through():
    p = PaperBroker()
    p.place(BUY_LONG, 0.48, 10)
    p.on_book(0.48, 0.50)                 # at the level: not cleared
    assert p.pos == 0
    p.on_book(0.46, 0.48)                 # bid dropped below us: filled
    assert p.pos == 10 and len(p.fills) == 1 and not p.open_orders()


def test_paper_pnl_and_settlement_both_ways():
    # buy Up at .48 then it settles Up: +0.52*10 + rebate;   settles Down: -0.48*10 + rebate
    for up, expect in ((True, 5.2), (False, -4.8)):
        p = PaperBroker()
        p.place(BUY_LONG, 0.48, 10)
        p.on_book(0.40, 0.45)
        rebate = 0.0125 * 0.48 * 0.52 * 10
        assert abs(p.settle(up) - (expect + rebate)) < 1e-9
        assert p.pos == 0


def test_paper_short_via_buy_short_accounts_like_selling_yes():
    p = PaperBroker()
    p.place(BUY_SHORT, 0.52, 10)          # buy NO == sell YES at .52
    p.on_book(0.55, 0.58)                 # ask side cleared upward
    assert p.pos == -10
    assert abs(p.settle(False) - (5.2 + 0.0125 * 0.52 * 0.48 * 10)) < 1e-9   # Down: NO wins, keep the .52
    p = PaperBroker(); p.place(BUY_SHORT, 0.52, 10); p.on_book(0.55, 0.58)
    assert abs(p.settle(True) - (5.2 - 10 + 0.0125 * 0.52 * 0.48 * 10)) < 1e-9  # Up: pay out $1 each


def test_rate_limiter_paces_requests():
    rl = RateLimiter(rps=20, burst=1)
    t = time.monotonic()
    for _ in range(5):
        rl.acquire()
    assert time.monotonic() - t >= 0.15    # 4 waits of ~50ms


def test_live_defaults_are_one_contract():
    d = HourlyConfig()
    assert d.size == 1.0 and d.max_pos <= 3.0


def test_up_and_down_are_exactly_symmetric():
    """P(Down) = 1 - P(Up): mirroring the state must mirror the quotes (nothing is Up-biased)."""
    from polymarket_mm.model import fair_up
    for d in (5, 40, 150, 400):
        up = fair_up(spot=84000 + d, k=84000, now=0, window_end=3000, sigma=6)
        dn = fair_up(spot=84000 - d, k=84000, now=0, window_end=3000, sigma=6)
        assert abs(up + dn - 1) < 1e-9
    a = {x.intent: x for x in yes_quotes(fair=0.7, half=0.02, tick=0.01, pos=0, cfg=CFG)}
    b = {x.intent: x for x in yes_quotes(fair=0.3, half=0.02, tick=0.01, pos=0, cfg=CFG)}
    # buying Up at x is the mirror of buying Down at x: bid(Up)@0.68 <-> ask side BUY_SHORT@0.32 -> Down price 0.68
    assert abs(a[BUY_LONG].price - (1 - b[BUY_SHORT].price)) < 1e-9
    assert abs(a[BUY_SHORT].price - (1 - b[BUY_LONG].price)) < 1e-9


def test_dead_even_market_quotes_dead_even_even_with_float_dust():
    for fair in (0.5, 0.5 + 4e-10, 0.5 - 4e-10):
        r = {x.intent: x for x in yes_quotes(fair=fair, half=0.02, tick=0.01, pos=0, cfg=CFG)}
        up_bid, down_bid = r[BUY_LONG].price, 1 - r[BUY_SHORT].price
        assert abs(up_bid - down_bid) < 1e-9, (fair, up_bid, down_bid)


def test_mirrored_fair_values_give_mirrored_quotes_across_the_range():
    for f in (0.07, 0.13, 0.26, 0.41, 0.5, 0.62, 0.78, 0.93):
        a = {x.intent: x for x in yes_quotes(fair=f, half=0.02, tick=0.01, pos=0, cfg=CFG)}
        b = {x.intent: x for x in yes_quotes(fair=1 - f, half=0.02, tick=0.01, pos=0, cfg=CFG)}
        assert abs(a[BUY_LONG].price - (1 - b[BUY_SHORT].price)) < 1e-9, f
        assert abs(a[BUY_SHORT].price - (1 - b[BUY_LONG].price)) < 1e-9, f


# ---- requoting follows fair value without churning -------------------------------------------
class _FakeBroker:
    def __init__(self, orders=()):
        self.orders = list(orders)
        self.placed, self.cancelled = [], []

    def open_orders(self):
        return list(self.orders)

    def place(self, intent, price, qty):
        self.placed.append((intent, price, qty))

    def cancel(self, oid):
        self.cancelled.append(oid)


def _maker(orders):
    from polymarket_mm.hourly_mm import HourlyMaker
    m = object.__new__(HourlyMaker)
    m.broker = _FakeBroker(orders)
    return m


def test_requote_when_fair_moves_enough():
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    m = _maker([RestingOrder("a", BUY_LONG, 0.48, 1.0)])
    # fair jumps to 0.60 -> desired bid 0.58; resting 0.48 is far off -> replace
    m._reconcile([DesiredQuote(BUY_LONG, 0.58, 1.0)], 0.01, fair=0.60, half=0.02)
    assert m.broker.cancelled == ["a"] and m.broker.placed == [(BUY_LONG, 0.58, 1.0)]


def test_no_churn_on_one_tick_wiggle_while_edge_is_intact():
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    m = _maker([RestingOrder("a", BUY_LONG, 0.48, 1.0)])
    # desired moved up one tick (0.49) but resting 0.48 still has 2c of edge vs fair 0.50 -> keep
    m._reconcile([DesiredQuote(BUY_LONG, 0.49, 1.0)], 0.01, fair=0.51, half=0.02)
    assert m.broker.cancelled == [] and m.broker.placed == []


def test_replaces_when_resting_order_has_lost_its_edge():
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    m = _maker([RestingOrder("a", BUY_LONG, 0.49, 1.0)])
    # fair fell to 0.495: resting bid 0.49 has ~0.5c edge (< half of 2c) -> must move down, not sit there
    m._reconcile([DesiredQuote(BUY_LONG, 0.48, 1.0)], 0.01, fair=0.495, half=0.02)
    assert m.broker.cancelled == ["a"] and m.broker.placed == [(BUY_LONG, 0.48, 1.0)]


def test_down_side_requotes_too():
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    m = _maker([RestingOrder("d", BUY_SHORT, 0.52, 1.0)])       # BID DOWN 0.48
    m._reconcile([DesiredQuote(BUY_SHORT, 0.42, 1.0)], 0.01, fair=0.40, half=0.02)   # fair Up fell -> Down bid rises
    assert m.broker.cancelled == ["d"] and m.broker.placed == [(BUY_SHORT, 0.42, 1.0)]


def test_order_body_gtd_expiry_format():
    from polymarket_mm.us_broker import order_body
    b = order_body("s", BUY_LONG, 0.48, 1, expire_at=1790786400)
    assert b["tif"] == "TIME_IN_FORCE_GOOD_TILL_DATE" and b["goodTillTime"].endswith("Z") and b["goodTillTime"].startswith("2026-")


class _FakeRest:
    def __init__(self):
        self.calls = []

    def call(self, method, path, params=None, body=None, priority=False):
        self.calls.append((method, path, body, priority))
        if path == "/v1/orders":
            return {"id": "X1"}
        if path == "/v1/orders/open":
            return {"orders": []}
        if path == "/v1/portfolio/positions":
            return {"positions": {}}
        if path == "/v1/account/balances":
            return {"balances": [{"buyingPower": 1.64}]}
        return {}


def test_live_cancel_sends_market_slug_and_skips_rate_limit_queue():
    """Regression: the exchange returns 400 unless the cancel body carries marketSlug."""
    from polymarket_mm.us_broker import LiveBroker
    r = _FakeRest()
    b = LiveBroker(r, "slug-1")
    b.cancel("ABC")
    assert r.calls[-1] == ("POST", "/v1/order/ABC/cancel", {"marketSlug": "slug-1"}, True)


def test_live_broker_respects_buying_power_and_tracks_orders_locally():
    from polymarket_mm.us_broker import LiveBroker
    r = _FakeRest()
    b = LiveBroker(r, "s")
    b.open_orders()                                    # primes buying power = 1.64
    assert b.place(BUY_LONG, 0.48, 1.0) == "X1"        # costs 0.48
    assert b.place(BUY_SHORT, 0.30, 1.0) == "X1"       # costs 0.70 -> total 1.18 <= 1.64
    assert b.place(BUY_LONG, 0.90, 1.0) is None        # 0.90 > remaining ~0.46: refused, no API call
    assert sum(1 for c in r.calls if c[1] == "/v1/orders") == 2


def test_broker_flags_fault_if_position_sign_contradicts_fills():
    from polymarket_mm.us_broker import LiveBroker, RestingOrder
    r = _FakeRest()
    b = LiveBroker(r, "s")
    b._orders = {"o1": RestingOrder("o1", BUY_LONG, 0.5, 1.0)}   # we bought 1 Up ...
    orig = r.call
    r.call = lambda m, p, params=None, body=None, priority=False: (
        {"positions": {"s": {"netPositionDecimal": "-1"}}} if p == "/v1/portfolio/positions" else orig(m, p, params, body, priority))
    b._refresh(force=True)                                        # ... order vanished, but position went to -1
    assert b.fault and "sign" in b.fault


def test_no_duplicate_orders_when_exchange_list_lags_behind_our_placement():
    """Regression (seen live): a fresh order wasn't in /orders/open yet; the old code forgot it and re-placed it."""
    from polymarket_mm.hourly_mm import DesiredQuote, HourlyMaker
    from polymarket_mm.us_broker import LiveBroker

    class LaggyRest(_FakeRest):          # exchange never lists anything (worst case lag)
        n = 0

        def call(self, method, path, params=None, body=None, priority=False):
            if path == "/v1/orders":
                LaggyRest.n += 1
                return {"id": f"O{LaggyRest.n}"}
            return super().call(method, path, params, body, priority)

    b = LiveBroker(LaggyRest(), "s", refresh=0.0)        # refresh on every read: the failure case
    m = object.__new__(HourlyMaker)
    m.broker = b
    want = [DesiredQuote(BUY_LONG, 0.44, 1.0), DesiredQuote(BUY_SHORT, 0.49, 1.0)]
    for _ in range(6):                                   # six loop iterations in a row
        m._reconcile(want, 0.01, fair=0.47, half=0.02)
    assert LaggyRest.n == 2, f"placed {LaggyRest.n} orders for 2 desired quotes"


def test_order_really_gone_after_grace_is_treated_as_filled():
    from polymarket_mm.us_broker import LiveBroker, RestingOrder
    r = _FakeRest()
    b = LiveBroker(r, "s")
    b._orders = {"o1": RestingOrder("o1", BUY_LONG, 0.5, 1.0)}
    b._placed_at = {"o1": time.monotonic() - 10}         # old enough: absent from the list means it's gone
    b._refresh(force=True)
    assert "o1" not in b._orders


def test_vanished_order_counts_as_filled_until_position_confirms():
    from polymarket_mm.us_broker import LiveBroker, RestingOrder
    r = _FakeRest()                                            # reports position 0 (lagging)
    b = LiveBroker(r, "s")
    b._orders = {"o1": RestingOrder("o1", BUY_LONG, 0.5, 1.0)}
    b._placed_at = {"o1": time.monotonic() - 10}
    b._refresh(force=True)                                     # order vanished, position still 0
    assert b.position() == 1.0                                 # assume the fill: cap can't be overshot
    b._pending = (1.0, time.monotonic() - 10)                  # 10s later with no confirmation: stop assuming
    assert b.position() == 0.0


# ---- stop the "yes loop": cap of 1, exit-only when holding, adverse-selection padding ----------------
def test_holding_one_contract_quotes_only_the_exit_no_stacking():
    cfg = HourlyConfig()
    assert cfg.max_pos == 1.0
    up = {x.intent: x for x in yes_quotes(fair=0.4, half=0.02, tick=0.01, pos=1.0, cfg=cfg)}
    assert set(up) == {SELL_LONG}                                   # no more Up bids while long Up
    dn = {x.intent: x for x in yes_quotes(fair=0.4, half=0.02, tick=0.01, pos=-1.0, cfg=cfg)}
    assert set(dn) == {SELL_SHORT}                                  # no more Down bids while long Down
    flat = {x.intent: x for x in yes_quotes(fair=0.4, half=0.02, tick=0.01, pos=0.0, cfg=cfg)}
    assert set(flat) == {BUY_LONG, BUY_SHORT}                       # flat -> two-sided entry


def test_exit_is_priced_to_actually_fill_when_holding():
    cfg = HourlyConfig()
    flat_ask = {x.intent: x for x in yes_quotes(fair=0.5, half=0.02, tick=0.01, pos=0.0, cfg=cfg)}[BUY_SHORT].price
    held_ask = {x.intent: x for x in yes_quotes(fair=0.5, half=0.02, tick=0.01, pos=1.0, cfg=cfg)}[SELL_LONG].price
    assert held_ask < flat_ask and held_ask <= 0.50                 # leans the exit toward/below fair to get out


def test_extra_half_after_fill_and_in_fast_markets():
    from polymarket_mm.hourly_mm import extra_half
    cfg = HourlyConfig()
    calm = [(100 + i, 0.50) for i in range(4)]
    assert extra_half(now=104, last_fill=0, fair_hist=calm, fair=0.50, cfg=cfg) == 0.0
    assert abs(extra_half(now=104, last_fill=90, fair_hist=calm, fair=0.50, cfg=cfg) - cfg.fill_widen) < 1e-12
    fast = [(101, 0.50), (102, 0.48), (103, 0.46)]
    assert abs(extra_half(now=104, last_fill=0, fair_hist=fast, fair=0.44, cfg=cfg) - 0.06) < 1e-9 or \
        abs(extra_half(now=104, last_fill=0, fair_hist=fast, fair=0.44, cfg=cfg) - cfg.trend_cap) < 1e-9
    assert extra_half(now=104, last_fill=0, fair_hist=fast, fair=0.0, cfg=cfg) == cfg.trend_cap   # capped


# ---- a network blip must not kill the bot (seen live: ProxyError ended the run and the cancel also failed) ----
def test_run_survives_transient_errors_and_keeps_going():
    from polymarket_mm.hourly_mm import HourlyMaker
    m = object.__new__(HourlyMaker)
    m.cfg, m.stop, m.broker = HourlyConfig(loop_hz=200.0), __import__("threading").Event(), None
    m._lock = __import__("threading").Lock()
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        if calls["n"] in (1, 2):
            raise ConnectionError("proxy blip")
        if calls["n"] >= 5:
            m.stop.set()

    m._step, m.shutdown, m._safe_cancel_all = step, lambda: None, lambda *a, **k: True
    m.run()
    assert calls["n"] >= 5                      # kept going after two errors


def test_run_gives_up_after_persistent_errors_and_still_shuts_down(monkeypatch):
    from polymarket_mm.hourly_mm import HourlyMaker
    m = object.__new__(HourlyMaker)
    m.cfg, m.stop, m.broker = HourlyConfig(max_errors=3), __import__("threading").Event(), None
    shut = []
    m._step = lambda: (_ for _ in ()).throw(ConnectionError("down"))
    m.shutdown, m._safe_cancel_all = lambda: shut.append(1), lambda *a, **k: True
    monkeypatch.setattr("polymarket_mm.hourly_mm.time.sleep", lambda s: None)   # don't actually wait (auto-restored)
    m.run()
    assert shut == [1]


def test_rest_retries_reads_but_never_retries_order_creation():
    import requests
    from polymarket_mm.us_broker import UsRest

    class Flaky:
        def __init__(self, fail):
            self.fail, self.calls = fail, 0

        def request(self, *a, **k):
            self.calls += 1
            if self.calls <= self.fail:
                raise requests.ConnectionError("blip")

            class R:
                status_code, ok, content = 200, True, b"{}"
                def json(self): return {}
            return R()

    class A:
        def headers(self, *a): return {}

    s = Flaky(2)
    UsRest(A(), rps=1000, session=s).call("GET", "/v1/orders/open")          # survives 2 blips
    assert s.calls == 3
    s2 = Flaky(1)
    try:
        UsRest(A(), rps=1000, session=s2).call("POST", "/v1/orders", body={})
        assert False, "order creation must not be retried"
    except requests.ConnectionError:
        pass
    assert s2.calls == 1


def test_exit_quote_priced_at_or_below_fair_is_not_replaced_every_cycle():
    """Regression (seen live: 46 placements in 25s): an exit deliberately priced at/below fair has ~0 'edge'."""
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    m = _maker([RestingOrder("x", SELL_LONG, 0.30, 1.0)])
    for _ in range(20):                                   # twenty loop iterations, fair hovering at the order's price
        m._reconcile([DesiredQuote(SELL_LONG, 0.30, 1.0)], 0.01, fair=0.301, half=0.02)
    assert m.broker.cancelled == [] and m.broker.placed == []


def test_resting_order_more_aggressive_than_desired_is_replaced_but_one_tick_passive_is_kept():
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    # ask-like: desired 0.30. resting 0.29 is MORE aggressive (gives edge away) -> replace
    m = _maker([RestingOrder("a", BUY_SHORT, 0.29, 1.0)])
    m._reconcile([DesiredQuote(BUY_SHORT, 0.30, 1.0)], 0.01, fair=0.30, half=0.02)
    assert m.broker.cancelled == ["a"]
    # resting 0.31 is one tick more passive -> keep
    m = _maker([RestingOrder("b", BUY_SHORT, 0.31, 1.0)])
    m._reconcile([DesiredQuote(BUY_SHORT, 0.30, 1.0)], 0.01, fair=0.30, half=0.02)
    assert m.broker.cancelled == [] and m.broker.placed == []
    # two ticks more passive -> replace
    m = _maker([RestingOrder("c", BUY_SHORT, 0.32, 1.0)])
    m._reconcile([DesiredQuote(BUY_SHORT, 0.30, 1.0)], 0.01, fair=0.30, half=0.02)
    assert m.broker.cancelled == ["c"]


def test_quoting_skips_the_weak_first_fifteen_minutes():
    from polymarket_mm.hourly_mm import HourlyMaker
    from polymarket_mm.hourly import HourlyMarket
    m = object.__new__(HourlyMaker)
    m.cfg = HourlyConfig()
    m.mkt = HourlyMarket("s", 10_000.0, 13_600.0, 84000.0, 0.01, 0.01, 0.0695, True)
    m.brti = type("B", (), {"latest": lambda self: object()})()
    m.book_feed = type("F", (), {"book": lambda self, s: object()})()
    m.sampler = type("S", (), {"px": {i: 1 for i in range(500)}, "window_avg": lambda self, t: 84000.0})()
    m._halt_until = 0.0
    assert m._quoteable(10_000 + 600) == "waiting for window open"      # minute 10: sit out
    assert m._quoteable(10_000 + 901) is None                            # minute 15+: quote
    assert m._quoteable(13_600 - 100) == "final minutes"                 # still stops before expiry


# ---- exits use the real book and stop chasing -------------------------------------------------------------
def test_exit_joins_the_best_ask_instead_of_quoting_above_the_book():
    # the live snapshot: Up book 0.41 / 0.42, model fair 0.433; we hold 1 Up
    cfg = HourlyConfig()
    no_book = {x.intent: x for x in yes_quotes(fair=0.433, half=0.02, tick=0.01, pos=1.0, cfg=cfg)}[SELL_LONG].price
    with_book = {x.intent: x for x in yes_quotes(fair=0.433, half=0.02, tick=0.01, pos=1.0, cfg=cfg, book=(0.41, 0.42))}[SELL_LONG].price
    assert abs(with_book - 0.42) < 1e-9 and with_book <= no_book      # joins the best ask


def test_exit_never_crosses_the_bid_and_never_dumps_below_fair_minus_slack():
    cfg = HourlyConfig()
    # book far below fair (book very bearish): don't give it away, wait at fair - slack
    p = {x.intent: x for x in yes_quotes(fair=0.60, half=0.02, tick=0.01, pos=1.0, cfg=cfg, book=(0.30, 0.31))}[SELL_LONG].price
    assert p >= 0.60 - cfg.exit_slack - 1e-9
    # a tight book: exit must stay strictly above the best bid
    p = {x.intent: x for x in yes_quotes(fair=0.40, half=0.02, tick=0.01, pos=1.0, cfg=cfg, book=(0.40, 0.41))}[SELL_LONG].price
    assert p > 0.40


def test_down_exit_mirrors_it():
    cfg = HourlyConfig()
    q = {x.intent: x for x in yes_quotes(fair=0.567, half=0.02, tick=0.01, pos=-1.0, cfg=cfg, book=(0.40, 0.41))}
    assert SELL_SHORT in q and q[SELL_SHORT].price < 0.41            # YES-price of selling Down: below the YES best ask


def test_sticky_exit_does_not_chase_a_slow_drift_but_entries_still_requote():
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    m = _maker([RestingOrder("x", SELL_LONG, 0.44, 1.0)])
    m._reconcile([DesiredQuote(SELL_LONG, 0.41, 1.0)], 0.01, fair=0.42, half=0.02)    # desired fell 3 ticks: hold
    assert m.broker.cancelled == [] and m.broker.placed == []
    m = _maker([RestingOrder("x", SELL_LONG, 0.46, 1.0)])
    m._reconcile([DesiredQuote(SELL_LONG, 0.41, 1.0)], 0.01, fair=0.42, half=0.02)    # 5 ticks: now re-price
    assert m.broker.cancelled == ["x"]
    m = _maker([RestingOrder("b", BUY_LONG, 0.40, 1.0)])
    m._reconcile([DesiredQuote(BUY_LONG, 0.43, 1.0)], 0.01, fair=0.45, half=0.02)     # entries stay tight (1 tick)
    assert m.broker.cancelled == ["b"]


# ---- two-sided flow: join the top of the book on BOTH sides -------------------------------------------------
def _entries(fair, book, tol=0.02, half=0.02):
    cfg = HourlyConfig()
    q = {x.intent: x for x in yes_quotes(fair=fair, half=half, tick=0.01, pos=0.0, cfg=cfg, book=book, touch_tol=tol)}
    return q[BUY_LONG].price, 1 - q[BUY_SHORT].price          # (Up bid, Down bid)


def test_live_snapshot_now_bids_down_at_the_top_of_the_down_book():
    """The situation you saw: fair Up 0.316 / Down 0.684, book Up 0.29/0.30 (Down 0.70/0.71)."""
    up_bid, down_bid = _entries(0.316, (0.29, 0.30))
    assert abs(up_bid - 0.29) < 1e-9
    assert abs(down_bid - 0.70) < 1e-9                        # was 0.66, four cents behind the Down book: never filled


def test_without_touch_mode_the_down_bid_stays_behind_the_book_old_behaviour():
    _, down_bid = _entries(0.316, (0.29, 0.30), tol=0.0)
    assert down_bid < 0.68


def test_touch_join_is_capped_by_the_model_not_unlimited():
    # book Down bid 0.75 is 6.6c above fair Down 0.684: too dear, so we do NOT chase it
    _, down_bid = _entries(0.316, (0.24, 0.25))
    assert down_bid <= 0.684 + 0.02 + 1e-9


def test_both_sides_are_symmetric_when_the_book_is_symmetric_around_fair():
    for f in (0.2, 0.35, 0.5, 0.65, 0.8):
        up, dn = _entries(f, (f - 0.005, f + 0.005))
        up2, dn2 = _entries(1 - f, (1 - f - 0.005, 1 - f + 0.005))
        assert abs(up - dn2) < 0.011 and abs(dn - up2) < 0.011


def test_touch_quotes_never_cross_the_book():
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        for bb, ba in ((f - 0.03, f - 0.02), (f + 0.02, f + 0.03), (f - 0.005, f + 0.005)):
            q = {x.intent: x for x in yes_quotes(fair=f, half=0.02, tick=0.01, pos=0.0, cfg=HourlyConfig(), book=(bb, ba), touch_tol=0.02)}
            if BUY_LONG in q:
                assert q[BUY_LONG].price < ba - 1e-9
            if BUY_SHORT in q:
                assert q[BUY_SHORT].price > bb + 1e-9


# ---- exits must actually exit: hold still, then cross ---------------------------------------------------------
def test_exit_take_up_and_down_and_guards():
    from polymarket_mm.hourly_mm import exit_take
    cfg = HourlyConfig()
    up = exit_take(pos=1.0, fair=0.30, best_bid=0.29, best_ask=0.30, tick=0.01, cfg=cfg)
    assert up == (SELL_LONG, 0.28, 1.0)                              # limit = best bid - 1 tick, worst price we accept
    dn = exit_take(pos=-1.0, fair=0.70, best_bid=0.29, best_ask=0.30, tick=0.01, cfg=cfg)
    assert dn == (SELL_SHORT, 0.31, 1.0)                             # mirror, in YES-price space
    assert exit_take(pos=0.0, fair=0.3, best_bid=0.29, best_ask=0.30, tick=0.01, cfg=cfg) is None
    # book far below fair (book very bearish vs model): model says hold -> don't dump
    assert exit_take(pos=1.0, fair=0.60, best_bid=0.29, best_ask=0.30, tick=0.01, cfg=cfg) is None
    assert exit_take(pos=-1.0, fair=0.30, best_bid=0.70, best_ask=0.71, tick=0.01, cfg=cfg) is None


def test_young_exit_is_not_chased_but_protects_itself_immediately():
    import time as _t
    from polymarket_mm.hourly_mm import DesiredQuote
    from polymarket_mm.us_broker import RestingOrder
    now = _t.monotonic()
    # fair fell 6 ticks: desired 0.35 but our young exit sits at 0.41 -> HOLD (do not chase down)
    m = _maker([RestingOrder("e", SELL_LONG, 0.41, 1.0, placed_at=now - 5)])
    m._reconcile([DesiredQuote(SELL_LONG, 0.35, 1.0)], 0.01, fair=0.36, half=0.02)
    assert m.broker.cancelled == [] and m.broker.placed == []
    # once it is old enough it may be re-priced
    m = _maker([RestingOrder("e", SELL_LONG, 0.41, 1.0, placed_at=now - 40)])
    m._reconcile([DesiredQuote(SELL_LONG, 0.35, 1.0)], 0.01, fair=0.36, half=0.02)
    assert m.broker.cancelled == ["e"]
    # fair ROSE: a young exit priced below desired is selling too cheap -> replace at once
    m = _maker([RestingOrder("e", SELL_LONG, 0.41, 1.0, placed_at=now - 5)])
    m._reconcile([DesiredQuote(SELL_LONG, 0.46, 1.0)], 0.01, fair=0.47, half=0.02)
    assert m.broker.cancelled == ["e"]


def test_paper_take_reduces_inventory_and_pays_the_taker_fee():
    from polymarket_mm.us_broker import PaperBroker
    b = PaperBroker()
    b.pos, b.cash = 1.0, -0.39                                        # bought 1 Up at 0.39
    b.take(SELL_LONG, 0.28, 1.0)
    assert b.pos == 0.0
    assert abs(b.cash - (-0.39 + 0.28 - 0.0695 * 0.28 * 0.72)) < 1e-9


def test_take_body_is_ioc_and_allowed_to_take():
    from polymarket_mm.us_broker import take_body
    b = take_body("s", SELL_LONG, 0.28, 1)
    assert b["tif"] == "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL" and b["participateDontInitiate"] is False and "goodTillTime" not in b
