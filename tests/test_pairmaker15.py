import pytest

from polymarket_mm.pairmaker15 import K, P, Leg, PairConfig, PairMaker, plan_entry
from polymarket_mm.us_broker import BUY_LONG, BUY_SHORT, PaperBroker
from polymarket_mm.us_feed import UsBook

CFG = PairConfig()


def book(bid, ask):
    return UsBook("x", bids=[(bid, 50)], asks=[(ask, 50)])


def test_aligned_books_give_a_one_cent_margin_pair():
    m, name, up, dn = plan_entry(book(0.30, 0.31), book(0.30, 0.31), 0.30, CFG)
    assert m == pytest.approx(0.01)
    assert up.intent == BUY_LONG and dn.intent == BUY_SHORT
    assert up.side_price + dn.side_price == pytest.approx(0.99)


def test_picks_the_pairing_with_the_bigger_margin_when_venues_are_crossed():
    # Kalshi dearer than Polymarket: Up is cheap on Polymarket, Down is cheap on Kalshi
    m, name, up, dn = plan_entry(book(0.32, 0.33), book(0.30, 0.31), 0.31, CFG)
    assert name == "Up@Polymarket+Down@Kalshi" and up.venue == P and dn.venue == K
    assert m == pytest.approx(0.03)
    # and the mirror image picks the other pairing: direction of the choice follows the prices
    m2, name2, up2, dn2 = plan_entry(book(0.30, 0.31), book(0.32, 0.33), 0.31, CFG)
    assert name2 == "Up@Kalshi+Down@Polymarket" and up2.venue == K and dn2.venue == P


def test_margin_rules():
    # exactly min_margin (1c) is allowed ...
    assert plan_entry(book(0.32, 0.33), book(0.32, 0.33), 0.32, CFG) is not None
    # ... a stricter requirement (2c) refuses the same books
    assert plan_entry(book(0.32, 0.33), book(0.32, 0.33), 0.32, PairConfig(min_margin=0.02)) is None
    # zero spread on both venues: margin 0 -> nothing to earn
    assert plan_entry(book(0.30, 0.30), book(0.30, 0.30), 0.30, CFG) is None
    # both pairings cost more than $1 (Up dearer on both): no pair
    assert plan_entry(book(0.41, 0.42), book(0.41, 0.42), 0.41, PairConfig(min_margin=0.05)) is None


def test_model_must_not_hate_either_leg():
    # fair Up 0.60: a Down leg at 0.69 is far above fair Down 0.40 + tol -> refuse
    assert plan_entry(book(0.30, 0.31), book(0.30, 0.31), 0.60, CFG) is None


def test_empty_book_returns_none():
    assert plan_entry(UsBook("k", bids=[], asks=[]), book(0.3, 0.31), 0.3, CFG) is None


def test_leg_side_price():
    assert Leg(K, BUY_LONG, 0.30, 1).side_price == pytest.approx(0.30)
    assert Leg(P, BUY_SHORT, 0.31, 1).side_price == pytest.approx(0.69)          # Down price = 1 - YES price


def make_maker():
    m = object.__new__(PairMaker)
    m.cfg = CFG
    m.brokers = {K: PaperBroker(maker_rebate=-0.0175, taker_fee=0.07), P: PaperBroker()}
    m._plan_legs, m._hedge_tries = {}, 0
    return m


def test_sync_orders_places_one_leg_per_venue_and_does_not_churn():
    m = make_maker()
    plan = plan_entry(book(0.30, 0.31), book(0.30, 0.31), 0.30, CFG)
    m._sync_orders(plan)
    ids = {v: [o.id for o in b.open_orders()] for v, b in m.brokers.items()}
    assert all(len(x) == 1 for x in ids.values())
    m._sync_orders(plan)                                                          # same plan again: nothing changes
    assert {v: [o.id for o in b.open_orders()] for v, b in m.brokers.items()} == ids
    m._sync_orders(None)                                                          # no plan: everything is withdrawn
    assert all(not b.open_orders() for b in m.brokers.values())


def test_both_legs_filling_locks_the_margin_whatever_the_outcome():
    m = make_maker()
    plan = plan_entry(book(0.30, 0.31), book(0.30, 0.31), 0.30, CFG)
    m._sync_orders(plan)
    m.brokers[K].on_book(0.28, 0.29)       # market trades down through the Up bid on whichever venue has it
    m.brokers[P].on_book(0.32, 0.33)       # and up through the Down bid (= YES ask) on the other
    assert abs(sum(b.pos for b in m.brokers.values())) < 1e-9                     # hedged: net zero exposure
    up_pnl = sum(b.settle(True) for b in m.brokers.values())
    m2 = make_maker(); m2._sync_orders(plan); m2.brokers[K].on_book(0.28, 0.29); m2.brokers[P].on_book(0.32, 0.33)
    down_pnl = sum(b.settle(False) for b in m2.brokers.values())
    assert up_pnl == pytest.approx(down_pnl)                                      # same profit either way
    assert 0.0 < up_pnl < 0.02                                                    # ~ the 1c margin, tiny fee effects


def test_lone_leg_is_exited_by_crossing_and_partner_order_cancelled():
    m = make_maker()
    m.brokers[K].pos, m.brokers[K].cash = 1.0, -0.30                               # long 1 Up on Kalshi, partner not filled
    m.brokers[P].place(BUY_SHORT, 0.31, 1.0)
    m._exit_lone_leg({K: 1.0, P: 0.0}, 0.30, book(0.29, 0.30), book(0.29, 0.30))
    assert m.brokers[K].pos == 0.0 and not m.brokers[P].open_orders()


def test_lone_leg_is_held_if_exit_would_dump_far_below_fair():
    m = make_maker()
    m.brokers[K].pos = 1.0
    m._exit_lone_leg({K: 1.0, P: 0.0}, 0.60, book(0.20, 0.21), book(0.20, 0.21))   # bid 0.20 vs fair 0.60: model says hold
    assert m.brokers[K].pos == 1.0


def test_log_formatting_handles_a_missing_side_of_the_book():
    from polymarket_mm.pairmaker15 import _f
    assert _f(None).strip() == "--" and _f(0.3) == "0.300"


# ---- money limits and directional loading -------------------------------------------------------------------
from polymarket_mm.pairmaker15 import choose_plan, direction_edge, leg_sizes, pair_cost, phase  # noqa: E402


def test_direction_edge_and_leg_sizes():
    kb, pb = book(0.38, 0.40), book(0.38, 0.40)                       # books mid 0.39
    assert direction_edge(kb, pb, 0.43) == pytest.approx(0.04)
    assert leg_sizes(0.04, CFG) == (2.0, 1.0)                          # model likes Up: load Up
    assert leg_sizes(-0.04, CFG) == (1.0, 2.0)                         # model likes Down: load Down
    assert leg_sizes(0.01, CFG) == (1.0, 1.0)                          # too close to call: no lean
    assert leg_sizes(None, CFG) == (1.0, 1.0)


def test_loaded_pair_keeps_the_direction_and_respects_the_margin():
    kb, pb = book(0.38, 0.40), book(0.38, 0.40)
    m, name, up, dn = choose_plan(kb, pb, 0.43, PairConfig(tol=0.10))
    assert up.qty == 2.0 and dn.qty == 1.0                             # Up favoured -> the Up leg is the big one
    assert m >= CFG.min_margin - 1e-9


def test_money_cap_trims_the_loaded_size_then_refuses():
    kb, pb = book(0.38, 0.40), book(0.38, 0.40)
    loaded = choose_plan(kb, pb, 0.43, PairConfig(tol=0.10, max_capital=10))
    assert pair_cost(loaded[2], loaded[3]) == pytest.approx(2 * 0.38 + 0.60)   # the loaded cost: 1.36
    tight = choose_plan(kb, pb, 0.43, PairConfig(tol=0.10, max_capital=1.05))
    assert tight[2].qty == 1.0 and tight[3].qty == 1.0                 # loaded size did not fit: falls back to base size
    assert pair_cost(tight[2], tight[3]) <= 1.05
    assert choose_plan(kb, pb, 0.43, PairConfig(tol=0.10, max_capital=0.50)) is None    # nothing fits: no trade


def test_phase_classification():
    assert phase({K: 0.0, P: 0.0}) == "flat"
    assert phase({K: 1.0, P: 0.0}) == "lone" and phase({K: 0.0, P: -1.0}) == "lone"
    assert phase({K: 2.0, P: -1.0}) == "paired"                        # a loaded pair is still 'paired': both legs filled


def test_kill_switch_trips_on_session_loss_paper():
    m = make_maker()
    m.cfg = PairConfig(max_loss=1.0)
    m.live, m.killed, m.total_pnl, m.pm = False, False, -0.4, None
    assert not m._check_loss(0.0) and not m.killed
    m.total_pnl = -1.2
    assert m._check_loss(0.0) and m.killed
    assert all(not b.open_orders() for b in m.brokers.values())


def test_live_mode_book_callback_never_touches_brokers():
    """Regression: the paper fill-simulation hook crashed on live brokers (no on_book) on every book update."""
    import threading
    m = object.__new__(PairMaker)
    m.live, m._lock, m.brokers = True, threading.Lock(), {K: object(), P: object()}     # live brokers have no on_book
    m._on_book(K)(book(0.3, 0.31))                                                        # must not raise
    m.live, m.brokers = False, {K: PaperBroker()}
    m._on_book(K)(book(0.3, 0.31))                                                        # paper path still works
    m.brokers = {}                                                                        # before brokers exist: no crash
    m._on_book(P)(book(0.3, 0.31))


def test_one_attempt_per_window_after_a_lone_leg_exit():
    """Seen live: after a lone-leg exit the bot immediately started another pair into the same move."""
    m = make_maker()
    m.window_done = False
    m.brokers[K].pos = 1.0
    m._exit_lone_leg({K: 1.0, P: 0.0}, 0.30, book(0.29, 0.30), book(0.29, 0.30))
    assert m.window_done is True


# ---- one leg filled -> buy the other side at once ---------------------------------------------------------------
def maker_with_legs(held_venue, held_intent, held_yes_price, partner_qty=1.0):
    m = make_maker()
    m.cfg = PairConfig()
    other = P if held_venue == K else K
    m._hedge_tries = 0
    m._plan_legs = {held_venue: Leg(held_venue, held_intent, held_yes_price, 1.0),
                    other: Leg(other, BUY_SHORT if held_intent == BUY_LONG else BUY_LONG, 0.5, partner_qty)}
    return m, other


def test_holding_up_buys_down_on_the_other_venue_when_the_pair_is_still_cheap():
    m, other = maker_with_legs(K, BUY_LONG, 0.40)                      # long Up on Kalshi at 0.40
    m.brokers[K].pos = 1.0
    pb = book(0.59, 0.60)                                               # Polymarket Down costs 1 - 0.59 = 0.41 -> combined 0.81? use realistic:
    pb = book(0.60, 0.61)                                               # Down = 0.40 -> combined 0.80 (well under $1.03): complete
    assert m._try_complete({K: 1.0, P: 0.0}, book(0.40, 0.41), pb)
    assert m.brokers[P].pos == -1.0                                     # now long Down on Polymarket: hedged, both venues hold a leg


def test_holding_down_buys_up_on_the_other_venue():
    m, other = maker_with_legs(P, BUY_SHORT, 0.60)                     # long Down on Polymarket at YES 0.60 => Down 0.40
    m.brokers[P].pos = -1.0
    assert m._try_complete({K: 0.0, P: -1.0}, book(0.58, 0.59), book(0.60, 0.61))   # Up on Kalshi costs 0.59: 0.40 + 0.59 = 0.99
    assert m.brokers[K].pos == 1.0


def test_does_not_complete_when_the_market_has_run_away():
    m, other = maker_with_legs(K, BUY_LONG, 0.40)                      # bought Up at 0.40, market then crashed
    m.brokers[K].pos = 1.0
    crashed = book(0.10, 0.11)                                          # Down now costs 0.90: combined 1.30 -> NOT worth completing
    assert not m._try_complete({K: 1.0, P: 0.0}, book(0.10, 0.11), crashed)
    assert m.brokers[P].pos == 0.0


def test_completion_is_attempted_once_and_partner_size_is_respected():
    m, other = maker_with_legs(K, BUY_LONG, 0.40, partner_qty=1.0)
    m.brokers[K].pos = 2.0                                              # a LOADED leg (2 Up)
    assert m._try_complete({K: 2.0, P: 0.0}, book(0.40, 0.41), book(0.60, 0.61))
    assert m.brokers[P].pos == -1.0                                     # hedges the partner size only: net +1 Up stays (the lean)
    m._hedge_tries = 3
    assert not m._try_complete({K: 2.0, P: -1.0}, book(0.40, 0.41), book(0.60, 0.61))   # out of tries



def test_hedge_limit_is_the_cost_cap_and_unfilled_hedge_is_not_treated_as_done():
    m, other = maker_with_legs(K, BUY_LONG, 0.40)
    m.brokers[K].pos = 1.0
    taken = []
    m.brokers[P].take = lambda intent, px, qty: taken.append((intent, px, qty)) or False      # exchange fills nothing
    pb = book(0.61, 0.62)                                   # Down costs 0.39: combined 0.79, well inside the cap
    assert not m._try_complete({K: 1.0, P: 0.0}, book(0.40, 0.41), pb)        # False: nothing actually filled
    assert taken and abs(taken[0][1] - (1 - (1.03 - 0.40))) < 1e-9            # limit = the price that makes the pair cost exactly 1.03
    assert m._hedge_tries == 1
    assert not m._try_complete({K: 1.0, P: 0.0}, book(0.40, 0.41), pb) and m._hedge_tries == 2   # it retries


def test_live_take_only_counts_real_fills():
    from polymarket_mm.kalshi_broker import KalshiLiveBroker
    from polymarket_mm.us_broker import LiveBroker

    class R:
        def __init__(self, resp): self.resp = resp
        def call(self, *a, **k): return self.resp

    kb = KalshiLiveBroker(R({"order_id": "x", "fill_count": "0.00"}), "T")
    assert kb.take(BUY_LONG, 0.5, 1.0) is False and kb._pending == (0.0, 0.0)          # IOC found nothing: not a fill
    kb = KalshiLiveBroker(R({"order_id": "x", "fill_count": "1.00"}), "T")
    assert kb.take(BUY_LONG, 0.5, 1.0) is True and kb._pending[0] == 1.0
    pb = LiveBroker(R({"id": "x", "executions": [{"type": "EXECUTION_TYPE_NEW", "lastShares": "0"}]}), "s")
    assert pb.take(BUY_SHORT, 0.5, 1.0) is False
    pb = LiveBroker(R({"id": "x", "executions": [{"type": "EXECUTION_TYPE_FILL", "lastShares": "1"}]}), "s")
    assert pb.take(BUY_SHORT, 0.5, 1.0) is True
