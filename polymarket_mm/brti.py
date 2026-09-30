"""Live BRTI (CF Benchmarks Bitcoin Real-Time Index) via Kalshi's `cfbenchmarks_value` websocket channel.

Kalshi settles BTC markets on the trailing 60-second average of BRTI, so we expose both the
latest 1s print (`spot`) and that average (`avg_60s`). Auth is Kalshi's signed-header scheme:
sign(timestamp_ms + "GET" + path) with your API private key (Ed25519 or RSA-PSS).
"""
import asyncio
import base64
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

log = logging.getLogger(__name__)
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"


@dataclass(frozen=True)
class BrtiTick:
    spot: float | None  # latest 1s BRTI print (None if the raw frame had no parsable value)
    avg_60s: float | None  # trailing 60s average = what Kalshi settles on
    avg_60s_ticks: int  # how many prints are in that average (60 = full window)
    quarter_hour_avg: float | None  # only present in the final minute before :00/:15/:30/:45
    received_at: float  # upstream receipt time, unix seconds
    local_ts: float  # when we got it, time.time()


class KalshiAuth:
    def __init__(self, key_id: str, private_key_pem: bytes):
        self.key_id = key_id
        self._key = serialization.load_pem_private_key(private_key_pem, password=None)

    @classmethod
    def from_env(cls) -> "KalshiAuth":
        key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
        path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
        pem = os.getenv("KALSHI_PRIVATE_KEY", "").replace("\\n", "\n").strip()
        if path:
            pem = Path(path).expanduser().read_text()
        if not key_id or not pem:
            raise SystemExit("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH (or KALSHI_PRIVATE_KEY) in .env")
        return cls(key_id, pem.encode())

    def sign(self, message: str) -> str:
        data = message.encode()
        if isinstance(self._key, ed25519.Ed25519PrivateKey):
            sig = self._key.sign(data)
        elif isinstance(self._key, rsa.RSAPrivateKey):
            sig = self._key.sign(
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        else:
            raise ValueError("unsupported key type (need Ed25519 or RSA)")
        return base64.b64encode(sig).decode()

    def headers(self, method: str = "GET", path: str = WS_PATH) -> dict:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self.sign(ts + method + path),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_message(raw: str | bytes, index_id: str = "BRTI") -> BrtiTick | None:
    """Parse one websocket frame; returns None for anything that isn't a value update for index_id."""
    try:
        m = json.loads(raw)
    except ValueError:
        return None
    if m.get("type") != "cfbenchmarks_value":
        return None
    msg = m.get("msg") or {}
    if msg.get("index_id") != index_id:
        return None

    # `data` is CF's raw frame as a JSON string; averages may sit there or beside it.
    frame = msg.get("data")
    if isinstance(frame, str):
        try:
            frame = json.loads(frame)
        except ValueError:
            frame = {}
    frame = frame if isinstance(frame, dict) else {}

    def find(key):
        return msg.get(key) if msg.get(key) is not None else frame.get(key)

    avg = find("avg_60s_data") or {}
    qh = find("last_60s_windowed_average_15min") or {}
    return BrtiTick(
        spot=_f(frame.get("value")),
        avg_60s=_f(avg.get("value")),
        avg_60s_ticks=int(avg.get("window_size") or 0),
        quarter_hour_avg=_f(qh.get("value")),
        received_at=(_f(msg.get("received_at")) or time.time() * 1000) / 1000,
        local_ts=time.time(),
    )


class BrtiFeed:
    """Background websocket client; call start(), then read `latest()` from any thread."""

    def __init__(self, auth: KalshiAuth, index_id: str = "BRTI", url: str = WS_URL, stale_after: float = 5.0):
        self.auth, self.index_id, self.url, self.stale_after = auth, index_id, url, stale_after
        self._tick: BrtiTick | None = None
        self._history: deque[tuple[float, float]] = deque(maxlen=600)  # (local_ts, price)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.on_tick = None  # optional callback(BrtiTick), runs on the feed thread

    def latest(self) -> BrtiTick | None:
        """Latest tick, or None if we have none or it's older than stale_after seconds."""
        with self._lock:
            t = self._tick
        if t is None or time.time() - t.local_ts > self.stale_after:
            return None
        return t

    def move(self, window: float) -> float | None:
        """Absolute fractional BRTI change over the last `window` seconds (None if not enough data)."""
        with self._lock:
            hist = list(self._history)
        if not hist:
            return None
        now_ts, now_px = hist[-1]
        old = [px for ts, px in hist if ts >= now_ts - window]
        if len(old) < 2 or now_ts - hist[0][0] < window / 2:
            return None
        return abs(now_px - old[0]) / old[0]

    def start(self) -> None:
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="brti-feed")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 1.0
            except Exception as e:
                log.warning("BRTI feed error: %s (reconnecting in %.0fs)", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _session(self) -> None:
        async with websockets.connect(self.url, additional_headers=self.auth.headers(), ping_interval=10) as ws:
            await ws.send(json.dumps({
                "id": 1, "cmd": "subscribe",
                "params": {"channels": ["cfbenchmarks_value"], "index_ids": [self.index_id]},
            }))
            log.info("BRTI feed connected")
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self.stale_after * 3)
                except asyncio.TimeoutError:
                    raise RuntimeError("no BRTI data, reconnecting")
                tick = parse_message(raw, self.index_id)
                if tick is None:
                    log.debug("non-tick frame: %.200s", raw)
                    continue
                px = tick.spot if tick.spot is not None else tick.avg_60s
                with self._lock:
                    self._tick = tick
                    if px:
                        self._history.append((tick.local_ts, px))
                if self.on_tick:
                    self.on_tick(tick)
