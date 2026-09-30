import json

import pytest

from polymarket_mm.hourly import candidate_slugs, parse_updown
from polymarket_mm.kalshi_feed import KalshiBook, SeqGap, apply_message, parse_market


def snap(seq=1, yes=(("0.4000", "10.00"), ("0.3900", "5.00")), no=(("0.5800", "7.00"), ("0.5700", "3.00"))):
    return json.dumps({"type": "orderbook_snapshot", "sid": 1, "seq": seq,
                       "msg": {"market_ticker": "T", "yes_dollars_fp": [list(x) for x in yes], "no_dollars_fp": [list(x) for x in no]}})


def delta(seq, side, price, d, ts=1790000000000):
    return json.dumps({"type": "orderbook_delta", "sid": 1, "seq": seq,
                       "msg": {"market_ticker": "T", "price_dollars": price, "delta_fp": d, "side": side, "ts_ms": ts}})


def test_snapshot_converts_to_yes_bids_and_asks():
    kb, st = KalshiBook("T"), {}
    assert apply_message(kb, snap(), st)
    b = kb.to_book()
    assert b.best_bid == 0.40 and b.best_ask == pytest.approx(0.42)       # YES ask = 1 - best NO bid (0.58)
    assert [p for p, _ in b.bids] == [0.40, 0.39] and [round(p, 2) for p, _ in b.asks] == [0.42, 0.43]
    assert b.spread == pytest.approx(0.02)


def test_deltas_add_reduce_and_remove_levels():
    kb, st = KalshiBook("T"), {}
    apply_message(kb, snap(), st)
    apply_message(kb, delta(2, "yes", "0.4100", "4.00"), st)               # new better bid
    assert kb.to_book().best_bid == 0.41
    apply_message(kb, delta(3, "yes", "0.4100", "-4.00"), st)              # fully removed
    assert kb.to_book().best_bid == 0.40
    apply_message(kb, delta(4, "no", "0.5800", "-7.00"), st)               # best NO bid gone -> ask moves up
    assert kb.to_book().best_ask == pytest.approx(0.43)
    apply_message(kb, delta(5, "yes", "0.4000", "-3.00"), st)              # partial reduction
    assert dict(kb.to_book().bids)[0.40] == 7.0


def test_sequence_gap_is_detected_not_ignored():
    kb, st = KalshiBook("T"), {}
    apply_message(kb, snap(seq=10), st)
    apply_message(kb, delta(11, "yes", "0.4100", "1.00"), st)
    with pytest.raises(SeqGap):
        apply_message(kb, delta(13, "yes", "0.4200", "1.00"), st)           # 12 was missed


def test_ignores_other_markets_junk_and_deltas_before_snapshot():
    kb, st = KalshiBook("T"), {}
    assert not apply_message(kb, delta(1, "yes", "0.4", "1"), st)           # no snapshot yet
    assert not apply_message(kb, "not json", st)
    other = json.dumps({"type": "orderbook_snapshot", "seq": 1, "msg": {"market_ticker": "OTHER", "yes_dollars_fp": [["0.5", "1"]]}})
    assert not apply_message(kb, other, st)
    assert not apply_message(kb, json.dumps({"type": "subscribed", "msg": {}}), st)


def test_empty_book_sides_are_safe():
    kb, st = KalshiBook("T"), {}
    apply_message(kb, snap(yes=(), no=()), st)
    b = kb.to_book()
    assert b.best_bid is None and b.best_ask is None and b.mid is None


def test_parse_kalshi_market():
    m = parse_market({"ticker": "KXBTC15M-26SEP301500-00", "strike_type": "greater_or_equal", "floor_strike": 83837.45,
                      "open_time": "2026-09-30T18:45:00Z", "close_time": "2026-09-30T19:00:00Z"})
    assert m.reference == 83837.45 and m.close_ts - m.open_ts == 900
    assert parse_market({"ticker": "x", "strike_type": "between", "floor_strike": 1}) is None
    assert parse_market({"ticker": "x"}) is None


def test_polymarket_15m_slugs_and_horizon_validation():
    import calendar
    t = calendar.timegm((2026, 9, 30, 18, 52, 10))
    assert candidate_slugs(t, "15m") == ["cpc-btc-updown-15m-2026-09-30-1845z", "cpc-btc-updown-15m-2026-09-30-1900z"]
    assert candidate_slugs(t, "1h")[0] == "cpc-btc-updown-1h-2026-09-30-1800z"      # hourly behaviour unchanged
    m = {"slug": "s", "orderPriceMinTickSize": 0.01, "active": True, "closed": False, "status": "MARKET_STATUS_OPEN",
         "assetPriceTerms": {"marketType": "ASSET_PRICE_MARKET_TYPE_UP_DOWN", "indexSymbol": "BRTI", "horizon": "15m",
                             "asset": {"symbol": "btc"}, "windowStart": "2026-09-30T18:45:00Z",
                             "windowEnd": "2026-09-30T19:00:00Z", "priceToBeat": {"value": "83837.45"}}}
    assert parse_updown(m, "15m").window_end - parse_updown(m, "15m").window_start == 900
    assert parse_updown(m, "1h") is None


def test_pair_costs_and_fees_math():
    from polymarket_mm.pairs15 import K_FEE, PM_FEE, fee, pair_costs
    from polymarket_mm.us_feed import UsBook
    pm = UsBook("p", bids=[(0.49, 10)], asks=[(0.51, 10)])
    ka = UsBook("k", bids=[(0.48, 10)], asks=[(0.50, 10)])
    c = pair_costs(pm, ka)
    a, b = c["A Up@Kalshi+Down@PM"], c["B Up@PM+Down@Kalshi"]
    assert a[0] == pytest.approx(0.50 + (1 - 0.49)) and b[0] == pytest.approx(0.51 + (1 - 0.48))   # 1.01 and 1.03
    assert a[1] == pytest.approx(fee(K_FEE, 0.50) + fee(PM_FEE, 0.51))
    assert a[2] == pytest.approx(a[0] + a[1]) and a[2] > 1                     # a normal pair is a LOSS after fees
    crossed = pair_costs(UsBook("p", bids=[(0.55, 1)], asks=[(0.56, 1)]), UsBook("k", bids=[(0.50, 1)], asks=[(0.52, 1)]))
    assert crossed["A Up@Kalshi+Down@PM"][0] < 1                               # PM bid 0.55 > Kalshi ask 0.52: gross < $1
    assert pair_costs(pm, UsBook("k", bids=[], asks=[])) is None
