import math

from polymarket_mm.hourly import parse_hourly
from polymarket_mm.model import SecondSampler, fair_up

END = 10_000.0


def test_at_the_money_far_out_is_half():
    p = fair_up(spot=84000, k=84000, now=END - 3000, window_end=END, sigma=6)
    assert abs(p - 0.5) < 0.01


def test_above_strike_is_favoured_and_monotone():
    ps = [fair_up(spot=84000 + d, k=84000, now=END - 1800, window_end=END, sigma=6) for d in (-500, -50, 0, 50, 500)]
    assert ps == sorted(ps) and ps[0] < 0.2 and ps[-1] > 0.8


def test_more_time_means_more_uncertainty():
    near = fair_up(spot=84050, k=84000, now=END - 120, window_end=END, sigma=6)
    far = fair_up(spot=84050, k=84000, now=END - 3000, window_end=END, sigma=6)
    assert near > far > 0.5


def test_after_expiry_is_deterministic_and_tie_goes_up():
    assert fair_up(spot=84000, k=84000, now=END + 1, window_end=END, sigma=6) == 1.0
    assert fair_up(spot=83999.99, k=84000, now=END + 1, window_end=END, sigma=6) == 0.0


def test_inside_final_minute_uses_known_prints():
    s = SecondSampler()
    for t in range(int(END) - 59, int(END) - 9):        # 50 known prints, all well above K
        s.add(t, 84100)
    p = fair_up(spot=84100, k=84000, now=END - 10, window_end=END, sigma=6, sampler=s)
    assert p > 0.99                                       # can't fall back below K with 10 prints left
    low = SecondSampler()
    for t in range(int(END) - 59, int(END) - 9):
        low.add(t, 83900)
    assert fair_up(spot=83900, k=84000, now=END - 10, window_end=END, sigma=6, sampler=low) < 0.01


def test_sampler_window_avg_and_sigma():
    s = SecondSampler()
    for t in range(1000, 1100):
        s.add(t, 100 + (t % 2))                          # tiny alternating noise
    assert abs(s.window_avg(1099) - 100.5) < 1e-9
    assert s.window_avg(5000) is None
    assert s.sigma(1099) >= 2.0                          # floor applies


def test_sigma_tracks_real_vol():
    import random
    random.seed(1)
    s, px = SecondSampler(), 84000.0
    for t in range(0, 900):
        px += random.gauss(0, 6)
        s.add(t, px)
    assert 4.5 < s.sigma(899) < 7.5


def test_parse_hourly_validates_by_fields():
    good = {"slug": "x", "orderPriceMinTickSize": 0.01, "minimumTradeQty": 0.01, "feeCoefficient": 0.0695,
            "active": True, "closed": False, "status": "MARKET_STATUS_OPEN",
            "assetPriceTerms": {"marketType": "ASSET_PRICE_MARKET_TYPE_UP_DOWN", "indexSymbol": "BRTI", "horizon": "1h",
                                "asset": {"symbol": "btc"}, "windowStart": "2026-09-30T16:00:00Z",
                                "windowEnd": "2026-09-30T17:00:00Z", "priceToBeat": {"value": "84088.21"}}}
    m = parse_hourly(good)
    assert m.price_to_beat == 84088.21 and m.window_end - m.window_start == 3600 and m.tradable
    assert parse_hourly({**good, "assetPriceTerms": {**good["assetPriceTerms"], "horizon": "15m"}}) is None
    assert parse_hourly({**good, "assetPriceTerms": {**good["assetPriceTerms"], "priceToBeat": None}}).price_to_beat is None
    assert parse_hourly({"slug": "sports"}) is None
