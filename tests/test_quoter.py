from polymarket_mm.config import Config
from polymarket_mm.quoter import compute_quotes

CFG = Config(quote_size=20, half_spread=0.01, skew=0.02, max_position=100)


def quotes(mid=0.50, yes_pos=0.0, no_pos=0.0, tick=0.01, min_size=5, cfg=CFG, **kw):
    return compute_quotes(
        yes_token="Y", no_token="N", mid=mid, tick=tick, min_size=min_size,
        yes_pos=yes_pos, no_pos=no_pos, cfg=cfg, **kw,
    )


def by(qs):
    return {(q.token_id, q.side): q for q in qs}


def test_flat_quotes_buy_yes_and_buy_no():
    q = by(quotes())
    assert q[("Y", "BUY")].price == 0.49 and q[("Y", "BUY")].size == 20
    assert q[("N", "BUY")].price == 0.49  # NO @ 1 - 0.51
    assert len(q) == 2


def test_never_crosses_itself():
    for mid in (0.2, 0.333, 0.5, 0.777):
        q = by(quotes(mid=mid))
        # YES bid + NO bid must total < 1 or a fill of both loses money
        assert q[("Y", "BUY")].price + q[("N", "BUY")].price < 1


def test_long_yes_sells_yes_and_skews_down():
    q = by(quotes(yes_pos=50))
    assert ("Y", "SELL") in q and ("N", "BUY") not in q
    assert q[("Y", "SELL")].size == 20
    flat = by(quotes())
    assert q[("Y", "BUY")].price < flat[("Y", "BUY")].price  # buys less aggressively


def test_sell_size_capped_by_inventory():
    q = by(quotes(yes_pos=8))
    assert q[("Y", "SELL")].size == 8


def test_holding_no_sells_no_on_bid_side():
    q = by(quotes(no_pos=30))
    assert ("N", "SELL") in q and ("Y", "BUY") not in q
    assert abs(q[("N", "SELL")].price - 0.51) < 1e-9  # 1 - bid(0.49)


def test_position_cap_stops_buying():
    q = by(quotes(yes_pos=3, no_pos=0, cfg=Config(quote_size=20, max_position=5)))
    # room for YES = 2 < min_size 5 -> no YES buy; still can quote NO
    assert ("Y", "BUY") not in q and ("N", "BUY") in q


def test_size_raised_to_min_order_size():
    q = by(quotes(cfg=Config(quote_size=2), min_size=5))
    assert q[("Y", "BUY")].size == 5


def test_extreme_price_returns_nothing():
    assert quotes(mid=0.01) == []
    assert quotes(mid=0.995) == []


def test_rewards_spread_clamps_half_spread():
    wide = Config(half_spread=0.05)
    q = by(quotes(cfg=wide, max_half_spread=0.02))
    assert q[("Y", "BUY")].price == 0.48


def test_fine_tick():
    q = by(quotes(mid=0.5, tick=0.001, cfg=Config(half_spread=0.0025)))
    assert q[("Y", "BUY")].price == 0.497
