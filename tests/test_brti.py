import base64
import json

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from polymarket_mm.brti import BrtiFeed, KalshiAuth, parse_message


def pem(key):
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


def test_ed25519_signature_verifies():
    k = ed25519.Ed25519PrivateKey.generate()
    a = KalshiAuth("kid", pem(k))
    h = a.headers()
    msg = h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/ws/v2"
    k.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg.encode())
    assert h["KALSHI-ACCESS-KEY"] == "kid"


def test_rsa_pss_signature_verifies():
    k = rsa.generate_private_key(65537, 2048)
    h = KalshiAuth("kid", pem(k)).headers()
    msg = h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/ws/v2"
    k.public_key().verify(
        base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256(),
    )


def frame(**over):
    m = {
        "type": "cfbenchmarks_value", "sid": 1, "seq": 3,
        "msg": {
            "index_id": "BRTI", "received_at": 1_780_000_000_500,
            "data": json.dumps({"value": "67123.45", "id": "BRTI"}),
            "avg_60s_data": {"value": "67100.12345678", "window_size": 42},
        },
    }
    m["msg"].update(over)
    return json.dumps(m)


def test_parse_tick():
    t = parse_message(frame())
    assert t.spot == 67123.45 and t.avg_60s == 67100.12345678 and t.avg_60s_ticks == 42
    assert t.quarter_hour_avg is None and t.received_at == 1_780_000_000.5


def test_parse_quarter_hour_and_other_index():
    t = parse_message(frame(last_60s_windowed_average_15min={"value": "67000.5", "window_size": 10}))
    assert t.quarter_hour_avg == 67000.5
    assert parse_message(frame(index_id="ETHUSD_RTI")) is None


def test_parse_ignores_junk():
    assert parse_message("not json") is None
    assert parse_message(json.dumps({"type": "subscribed", "msg": {}})) is None
    assert parse_message(frame(data="garbage")).spot is None  # still yields the averages


def test_latest_goes_stale():
    f = BrtiFeed(KalshiAuth("k", pem(ed25519.Ed25519PrivateKey.generate())), stale_after=5)
    assert f.latest() is None
    f._tick = parse_message(frame())
    assert f.latest() is not None
    f._tick = type(f._tick)(**{**f._tick.__dict__, "local_ts": f._tick.local_ts - 10})
    assert f.latest() is None


def test_move_over_window():
    f = BrtiFeed(KalshiAuth("k", pem(ed25519.Ed25519PrivateKey.generate())))
    assert f.move(30) is None
    for i, px in enumerate([100.0, 100.0, 101.0]):
        f._history.append((1000.0 + i * 20, px))  # 40s of data
    assert abs(f.move(30) - 0.01) < 1e-9


def frame5(value="84099.71", **over):
    m = {"type": "cfbenchmarks_value_5hz", "sid": 2, "seq": 1,
         "msg": {"index_id": "BRTI", "value_usd": value, "source_ts_ms": 1, "received_at": 1_780_000_000_200,
                 "data": json.dumps({"type": "value", "time": 1, "id": "BRTI", "value": value})}}
    m["msg"].update(over)
    return json.dumps(m)


def test_parse_5hz():
    t = parse_message(frame5())
    assert t.fast and t.spot == 84099.71 and t.avg_60s is None


def test_merge_keeps_5hz_spot_and_1hz_average():
    f = BrtiFeed(KalshiAuth("k", pem(ed25519.Ed25519PrivateKey.generate())))
    f._merge(parse_message(frame()))                       # 1Hz first: has the average
    m = f._merge(parse_message(frame5("70000.5")))          # 5Hz: newer spot, average retained
    assert m.spot == 70000.5 and m.avg_60s == 67100.12345678 and m.fast
    m = f._merge(parse_message(frame()))                    # late 1Hz must not overwrite the fresher 5Hz spot
    assert m.spot == 70000.5 and m.avg_60s == 67100.12345678
