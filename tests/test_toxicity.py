import json
import threading

import pytest

from polymarket_mm.toxicity import (MarkoutTracker, ToxicityGate, TradeFlow, depth_imbalance, micro_skew, parse_trade,
                                    pressure)
from polymarket_mm.us_feed import UsBook

# the live snapshot from the conversation: Up 0.10 x 79 / 0.11 x 579 -> lots of size offered, little bid: down pressure
ASK_HEAVY = UsBook("s", bids=[(0.10, 79.0), (0.09, 40.0), (0.08, 20.0)], asks=[(0.11, 579.0), (0.12, 300.0), (0.13, 100.0)])
BID_HEAVY = UsBook("s", bids=[(0.10, 579.0), (0.09, 300.0), (0.08, 100.0)], asks=[(0.11, 79.0), (0.12, 40.0), (0.13, 20.0)])
BALANCED = UsBook("s", bids=[(0.10, 100.0), (0.09, 100.0)], asks=[(0.11, 100.0), (0.12, 100.0)])

REAL_FRAME = json.dumps({"requestId": "t1", "subscriptionType": "SUBSCRIPTION_TYPE_TRADE", "trade": {
    "marketSlug": "m", "price": {"value": "0.0300", "currency": "USD"}, "quantity": {"value": "25.0000", "currency": "USD"},
    "tradeTime": "2026-09-30T19:49:59.788790208Z", "maker": {"side": "ORDER_SIDE_BUY", "intent": "ORDER_INTENT_BUY_LONG"}}})


def test_depth_imbalance_and_microprice_point_the_same_way():
    assert depth_imbalance(ASK_HEAVY) < -0.5 and micro_skew(ASK_HEAVY) < -0.5        # ask-heavy: price leans DOWN
    assert depth_imbalance(BID_HEAVY) > 0.5 and micro_skew(BID_HEAVY) > 0.5          # bid-heavy: leans UP
    assert abs(depth_imbalance(BALANCED)) < 1e-9 and abs(micro_skew(BALANCED)) < 1e-9
    assert micro_skew(UsBook("s", bids=[], asks=[])) == 0.0


def test_parse_real_trade_frame():
    px, qty, sign, ts = parse_trade(REAL_FRAME)
    assert (px, qty, sign) == (0.03, 25.0, -1)                      # maker BUY => a seller hit the bid => down pressure
    assert ts == pytest.approx(1790797799.79, abs=0.01)
    sell = json.loads(REAL_FRAME); sell["trade"]["maker"]["side"] = "ORDER_SIDE_SELL"
    assert parse_trade(json.dumps(sell))[2] == +1                   # maker SELL => a buyer lifted the ask
    assert parse_trade("junk") is None and parse_trade('{"marketData":{}}') is None


def test_trade_flow_window_and_minimum_volume():
    f = TradeFlow(window=10, min_volume=5)
    f.add(100.0, 2.0, -1)
    assert f.imbalance(101.0) == 0.0                                # too little volume to trust
    f.add(101.0, 10.0, -1); f.add(102.0, 2.0, +1)
    assert f.imbalance(103.0) == pytest.approx((-2 - 10 + 2) / 14)  # mostly sellers
    assert f.imbalance(120.0) == 0.0                                # everything aged out


def test_pressure_combines_the_three_signals():
    assert pressure(ASK_HEAVY, -1.0) < -0.7
    assert pressure(BID_HEAVY, +1.0) > 0.7
    assert abs(pressure(BALANCED, 0.0)) < 1e-9
    assert -0.6 < pressure(ASK_HEAVY, +0.9) < 0.3                   # heavy buying offsets a thin book


def test_gate_trips_holds_with_hysteresis_then_cools_down():
    g = ToxicityGate(trip=0.45, rearm=0.25, cooldown=3.0)
    assert g.update(0.0, -0.2) == (False, False, False, False)
    bu, bd, nd, nu = g.update(1.0, -0.5)                            # down pressure trips: block UP bids, flag the new trip
    assert bu and not bd and nd and not nu
    assert g.update(2.0, -0.35)[0] is True                          # between trip and re-arm: STILL blocked (hysteresis)
    assert g.update(3.0, -0.1)[0] is True                           # cleared, but cooling down for 3s
    assert g.update(5.0, -0.1)[0] is True
    assert g.update(6.5, -0.1)[0] is False                          # cooldown over
    bu, bd, nd, nu = g.update(7.0, +0.6)                            # up pressure: block DOWN bids, not Up bids
    assert bd and not bu and nu


def test_markout_tracker_judges_fills_after_the_horizon():
    m = MarkoutTracker(horizon=10, alpha=0.5)
    m.add(0.0, +1, 0.30)                                            # bought Up at 0.30
    assert m.update(5.0, 0.25) == []                                # too early
    assert m.update(10.0, 0.27) == [pytest.approx(-0.03)]           # mid fell to 0.27: adverse
    assert m.ewma == pytest.approx(-0.03) and m.n == 1
    m.add(11.0, -1, 0.60)                                           # sold Up (long Down) at 0.60; mid then 0.55: good for us
    assert m.update(21.0, 0.55)[0] == pytest.approx(0.05)
    assert m.ewma == pytest.approx(0.5 * -0.03 + 0.5 * 0.05)


# ---- integration: the maker pulls the endangered entry at once and keeps the OTHER side ------------------------------
def make_maker():
    from polymarket_mm.hourly_mm import HourlyConfig, HourlyMaker
    from polymarket_mm.us_broker import PaperBroker
    m = object.__new__(HourlyMaker)
    m.cfg, m.broker, m._lock = HourlyConfig(), PaperBroker(), threading.Lock()
    m.gate = ToxicityGate(); m.flow = TradeFlow(); m._pressure = 0.0; m._pulls = 0
    return m


def test_down_pressure_pulls_the_up_bid_immediately_and_leaves_the_down_bid():
    from polymarket_mm.us_broker import BUY_LONG, BUY_SHORT
    m = make_maker()
    m.broker.place(BUY_LONG, 0.10, 1.0)        # Up bid
    m.broker.place(BUY_SHORT, 0.11, 1.0)       # Down bid
    m._on_trade(REAL_FRAME)                    # a seller hits the bid
    m.flow.add(__import__("time").time(), 50.0, -1)
    m._on_book(ASK_HEAVY)                      # and the book is ask-heavy
    left = {o.intent for o in m.broker.open_orders()}
    assert BUY_LONG not in left and BUY_SHORT in left and m._pulls == 1


def test_up_pressure_pulls_the_down_bid_not_the_up_bid():
    from polymarket_mm.us_broker import BUY_LONG, BUY_SHORT
    m = make_maker()
    m.broker.place(BUY_LONG, 0.10, 1.0); m.broker.place(BUY_SHORT, 0.11, 1.0)
    m.flow.add(__import__("time").time(), 50.0, +1)
    m._on_book(BID_HEAVY)
    left = {o.intent for o in m.broker.open_orders()}
    assert BUY_SHORT not in left and BUY_LONG in left


def test_balanced_book_pulls_nothing():
    from polymarket_mm.us_broker import BUY_LONG, BUY_SHORT
    m = make_maker()
    m.broker.place(BUY_LONG, 0.10, 1.0); m.broker.place(BUY_SHORT, 0.11, 1.0)
    m._on_book(BALANCED)
    assert len(m.broker.open_orders()) == 2 and m._pulls == 0
