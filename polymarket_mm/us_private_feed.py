"""Polymarket US PRIVATE websocket (wss://api.polymarket.us/v1/ws/private): pushes our own order and position updates.

We use it only as a doorbell: any order/position frame calls `on_event`, which makes the broker re-read the truth from REST
at once instead of waiting for its ~1s polling timer. REST stays the source of truth, so a missed or odd frame cannot
corrupt state; the worst case is the old polling latency.
"""
import asyncio
import json
import logging
import threading
import time

import websockets

from .us_feed import UsAuth

log = logging.getLogger(__name__)
PRIVATE_URL = "wss://api.polymarket.us/v1/ws/private"
PRIVATE_PATH = "/v1/ws/private"


class UsPrivateFeed:
    def __init__(self, auth: UsAuth, slugs: list[str], url: str = PRIVATE_URL):
        self.auth, self.slugs, self.url = auth, list(slugs), url
        self.on_event = None                      # callback(frame_dict), on the feed thread
        self._stop = threading.Event()
        self.stats = {"frames": 0, "events": 0, "reconnects": 0}
        self.last_frames: list[str] = []          # a few raw frames, for debugging

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="us-private").start()

    def stop(self) -> None:
        self._stop.set()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 1.0
            except Exception as e:
                self.stats["reconnects"] += 1
                log.warning("US private feed error: %s (retry in %.0fs)", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _session(self) -> None:
        async with websockets.connect(self.url, additional_headers=self.auth.headers("GET", PRIVATE_PATH),
                                      ping_interval=5, ping_timeout=5) as ws:
            for typ in ("SUBSCRIPTION_TYPE_ORDER", "SUBSCRIPTION_TYPE_POSITION"):
                await ws.send(json.dumps({"subscribe": {"requestId": f"{typ[18:].lower()}-{int(time.time())}",
                                                        "subscriptionType": typ, "marketSlugs": self.slugs}}))
            log.info("US private feed connected (orders + positions for %d market)", len(self.slugs))
            async for raw in ws:
                if self._stop.is_set():
                    return
                self.stats["frames"] += 1
                if len(self.last_frames) < 6:
                    self.last_frames.append(raw if isinstance(raw, str) else raw.decode(errors="replace"))
                try:
                    j = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(j, dict) or "heartbeat" in j:
                    continue
                self.stats["events"] += 1
                if self.on_event:
                    try:
                        self.on_event(j)
                    except Exception:
                        log.exception("on_event failed")
