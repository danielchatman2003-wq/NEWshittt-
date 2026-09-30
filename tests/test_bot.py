from types import SimpleNamespace as NS

from polymarket_mm.bot import MarketMaker
from polymarket_mm.broker import DryRunBroker
from polymarket_mm.config import Config
from polymarket_mm.markets import Market

M = Market("Q?", "0xabc", "Y", "N", 0.5, 5, 0.01, None, 1e5)


class FakeClob:
    def __init__(self, bid="0.49", ask="0.51"):
        self.bid, self.ask = bid, ask

    def get_order_book(self, token):
        bids = [NS(price="0.10", size="9")] + ([NS(price=self.bid, size="9")] if self.bid else [])
        asks = [NS(price="0.90", size="9")] + ([NS(price=self.ask, size="9")] if self.ask else [])
        return NS(bids=bids, asks=asks, tick_size="0.01", min_order_size="5")


def make(clob=None, cfg=None):
    b = DryRunBroker()
    return MarketMaker(cfg or Config(), clob or FakeClob(), b, [M]), b


def test_places_then_leaves_orders_alone():
    bot, b = make()
    bot.cycle()
    ids = {o["id"] for o in b.open_orders("0xabc")}
    assert len(ids) == 2
    bot.cycle()
    assert {o["id"] for o in b.open_orders("0xabc")} == ids  # no churn


def test_requotes_when_mid_moves_a_little():
    bot, b = make()
    bot.cycle()
    before = {o["id"] for o in b.open_orders("0xabc")}
    bot.clob.bid, bot.clob.ask = "0.51", "0.53"
    bot.cycle()
    after = {o["id"] for o in b.open_orders("0xabc")}
    assert len(after) == 2 and not (before & after)


def test_big_jump_pulls_quotes():
    bot, b = make()
    bot.cycle()
    bot.clob.bid, bot.clob.ask = "0.59", "0.61"
    bot.cycle()
    assert b.open_orders("0xabc") == []


def test_wide_or_empty_book_pulls_quotes():
    bot, b = make()
    bot.cycle()
    bot.clob.bid = None
    bot.cycle()
    assert b.open_orders("0xabc") == []


def test_notional_cap():
    bot, b = make(cfg=Config(max_open_notional=12))  # one 20-share @0.49 order costs 9.8
    bot.cycle()
    assert len(b.open_orders("0xabc")) == 1


def test_shutdown_cancels_everything():
    bot, b = make()
    bot.cycle()
    bot.shutdown()
    assert b.open_orders("0xabc") == []


def test_repeated_failures_exit_nonzero():
    class Boom(FakeClob):
        def get_order_book(self, t):
            raise RuntimeError("api down")

    bot, b = make(clob=Boom(), cfg=Config(max_errors=2, refresh_seconds=0))
    bot._refresh_markets = lambda: None
    assert bot.run() == 1


def test_run_selects_markets_on_first_iteration(monkeypatch):
    """Regression: selection must not depend on time.monotonic() (uptime) being large."""
    monkeypatch.setattr("polymarket_mm.bot.time.monotonic", lambda: 5.0)
    calls = []
    monkeypatch.setattr("polymarket_mm.bot.select_markets", lambda cfg: calls.append(1) or [M])
    bot, b = make()
    bot.markets = []
    placed = []
    # stop after the first cycle, remembering what was resting at that point
    bot.stop.wait = lambda _s: (placed.extend(b.open_orders("0xabc")), bot.stop.set())
    assert bot.run() == 0
    assert calls == [1] and len(placed) == 2


class FakeBrti:
    def __init__(self, tick=object(), move=None):
        self.tick, self._move = tick, move

    def latest(self):
        return self.tick

    def move(self, window):
        return self._move


BTC = Market("Bitcoin above $70k?", "0xbtc", "Y", "N", 0.5, 5, 0.01, None, 1e5)


def btc_bot(brti):
    b = DryRunBroker()
    return MarketMaker(Config(), FakeClob(), b, [BTC, M], brti=brti), b


def test_btc_market_quotes_when_brti_calm():
    bot, b = btc_bot(FakeBrti(move=0.0002))
    bot.cycle()
    assert len(b.open_orders("0xbtc")) == 2


def test_btc_market_pulled_on_brti_spike_but_others_unaffected():
    bot, b = btc_bot(FakeBrti(move=0.0002))
    bot.cycle()
    bot.brti._move = 0.01
    bot.cycle()
    assert b.open_orders("0xbtc") == [] and len(b.open_orders("0xabc")) == 2


def test_btc_market_pulled_when_feed_stale():
    bot, b = btc_bot(FakeBrti(tick=None))
    bot.cycle()
    assert b.open_orders("0xbtc") == [] and len(b.open_orders("0xabc")) == 2
