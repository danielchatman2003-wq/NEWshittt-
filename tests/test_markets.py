import json
from datetime import datetime, timezone

from polymarket_mm.config import Config
from polymarket_mm.markets import parse_market

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


def raw(**over):
    m = {
        "question": "Q?", "conditionId": "0xabc", "active": True, "closed": False,
        "enableOrderBook": True, "acceptingOrders": True,
        "clobTokenIds": json.dumps(["1", "2"]), "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.6", "0.4"]), "volume24hr": 50000,
        "orderMinSize": 5, "orderPriceMinTickSize": 0.01,
        "rewardsMaxSpread": 2.5, "endDate": "2027-01-01T00:00:00Z",
    }
    m.update(over)
    return m


def test_accepts_good_market():
    m = parse_market(raw(), Config(), NOW)
    assert m and m.yes_token == "1" and m.no_token == "2"
    assert m.rewards_max_spread == 0.025


def test_rejections():
    cfg = Config()
    assert parse_market(raw(closed=True), cfg, NOW) is None
    assert parse_market(raw(acceptingOrders=False), cfg, NOW) is None
    assert parse_market(raw(volume24hr=10), cfg, NOW) is None
    assert parse_market(raw(outcomePrices=json.dumps(["0.97", "0.03"])), cfg, NOW) is None
    assert parse_market(raw(outcomes=json.dumps(["Trump", "Biden"])), cfg, NOW) is None
    assert parse_market(raw(endDate="2026-10-01T00:00:00Z"), cfg, NOW) is None
    assert parse_market({"question": "broken"}, cfg, NOW) is None


def test_no_rewards():
    assert parse_market(raw(rewardsMaxSpread=None), Config(), NOW).rewards_max_spread is None
