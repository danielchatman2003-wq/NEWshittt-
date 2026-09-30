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
