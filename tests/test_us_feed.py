import base64
import json
import time

from cryptography.hazmat.primitives.asymmetric import ed25519

from polymarket_mm.us_feed import UsAuth, UsBookFeed, parse_frame

FRAME = {"marketData": {
    "marketSlug": "m1",
    "bids": [{"px": {"value": "0.540", "currency": "USD"}, "qty": "10"}, {"px": {"value": "0.555", "currency": "USD"}, "qty": "0.50"}],
    "offers": [{"px": {"value": "0.570", "currency": "USD"}, "qty": "3"}, {"px": {"value": "0.560", "currency": "USD"}, "qty": "0.80"}],
    "state": "MARKET_STATE_OPEN",
    "stats": {"lastTradePx": {"value": "0.55", "currency": "USD"}},
    "transactTime": "2024-01-15T10:30:00Z",
}}


def test_parse_sorts_book_and_computes_top():
    b = parse_frame(json.dumps(FRAME))
    assert b.slug == "m1" and b.best_bid == 0.555 and b.best_ask == 0.56
    assert [p for p, _ in b.bids] == [0.555, 0.54] and [p for p, _ in b.asks] == [0.56, 0.57]
    assert abs(b.mid - 0.5575) < 1e-9 and abs(b.spread - 0.005) < 1e-9
    assert b.last_trade == 0.55 and b.exchange_ts == 1705314600.0


def test_parse_ignores_non_book_frames():
    for f in ('{"heartbeat":{}}', "junk", '{"trade":{"marketSlug":"m1"}}', '{"marketData":{}}'):
        assert parse_frame(f) is None


def test_one_sided_and_empty_books():
    b = parse_frame(json.dumps({"marketData": {"marketSlug": "m", "bids": [], "offers": [{"px": {"value": "0.5"}, "qty": "1"}]}}))
    assert b.best_bid is None and b.mid is None and b.best_ask == 0.5


def test_auth_signature_verifies():
    k = ed25519.Ed25519PrivateKey.generate()
    seed = k.private_bytes_raw()
    a = UsAuth("kid", base64.b64encode(seed + k.public_key().public_bytes_raw()).decode())  # 64-byte secret form
    h = a.headers()
    k.public_key().verify(base64.b64decode(h["X-PM-Signature"]), (h["X-PM-Timestamp"] + "GET/v1/ws/markets").encode())
    assert h["X-PM-Access-Key"] == "kid"
    # a bare 32-byte secret works too
    UsAuth("kid", base64.b64encode(seed).decode()).headers()


def test_book_goes_stale_when_feed_quiet():
    k = ed25519.Ed25519PrivateKey.generate()
    f = UsBookFeed(UsAuth("k", base64.b64encode(k.private_bytes_raw()).decode()), ["m1"], stale_after=5)
    assert f.book("m1") is None
    f._books["m1"] = parse_frame(json.dumps(FRAME))
    f._last_frame = time.time()
    assert f.book("m1") is not None
    f._last_frame = time.time() - 10
    assert f.book("m1") is None


def test_session_end_to_end_against_local_server():
    """Real websocket round trip: auth headers, subscribe payload, push updates, reconnect."""
    import asyncio
    import threading

    import websockets

    seen = {"headers": [], "subs": []}

    async def handler(ws):
        seen["headers"].append(dict(ws.request.headers))
        seen["subs"].append(json.loads(await ws.recv()))
        await ws.send('{"heartbeat":{}}')
        await ws.send(json.dumps(FRAME))
        if len(seen["subs"]) == 1:
            await ws.close()  # force a reconnect
        else:
            await asyncio.sleep(2)

    ready, port = threading.Event(), {}

    def serve():
        async def main():
            async with websockets.serve(handler, "127.0.0.1", 0) as s:
                port["p"] = s.sockets[0].getsockname()[1]
                ready.set()
                await asyncio.sleep(6)
        asyncio.run(main())

    threading.Thread(target=serve, daemon=True).start()
    ready.wait(5)

    k = ed25519.Ed25519PrivateKey.generate()
    feed = UsBookFeed(UsAuth("kid", base64.b64encode(k.private_bytes_raw()).decode()), ["m1"], url=f"ws://127.0.0.1:{port['p']}")
    got = []
    feed.on_update = got.append
    feed.start()
    deadline = time.time() + 5
    while len(got) < 2 and time.time() < deadline:
        time.sleep(0.05)
    feed.stop()

    assert len(got) >= 2, "expected an update before AND after the forced reconnect"
    assert seen["headers"][0]["x-pm-access-key"] == "kid" and "x-pm-signature" in seen["headers"][0]
    sub = seen["subs"][0]["subscribe"]
    assert sub["subscriptionType"] == "SUBSCRIPTION_TYPE_MARKET_DATA" and sub["marketSlugs"] == ["m1"]
    assert sub["responsesDebounced"] is False
    assert feed.book("m1").best_bid == 0.555
