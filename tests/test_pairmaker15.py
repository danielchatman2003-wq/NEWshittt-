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
