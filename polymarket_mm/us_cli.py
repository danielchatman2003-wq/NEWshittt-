"""Watch Polymarket US books live:  python -m polymarket_mm.us_cli --slug <market-slug> [-v]
Find slugs with:                    python -m polymarket_mm.us_cli --discover"""
import argparse
import logging
import time

import requests
from dotenv import load_dotenv

from .us_feed import REST, UsAuth, UsBookFeed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", default=[])
    ap.add_argument("--discover", action="store_true", help="list some active market slugs and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every non-book frame")
    args = ap.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    auth = UsAuth.from_env()

    if args.discover:
        r = requests.get(f"{REST}/v1/markets", params={"active": "true", "closed": "false", "limit": 30}, headers=auth.headers("GET", "/v1/markets"), timeout=15)
        print("HTTP", r.status_code)
        if not r.ok:
            print(r.text[:500])
            return
        for m in r.json().get("markets", []):
            print(f"{m['slug']:<48} {m.get('question', '')[:60]}")
        return
    if not args.slug:
        ap.error("pass --slug (or --discover)")

    feed = UsBookFeed(auth, args.slug)

    def show(b):
        age = f"{(b.local_ts - b.exchange_ts) * 1000:.0f}ms" if b.exchange_ts else "n/a"
        print(f"{b.slug}  bid={b.best_bid} ask={b.best_ask} mid={b.mid} spread={b.spread}  "
              f"depth={len(b.bids)}x{len(b.asks)}  state={b.state}  last_change_age={age}", flush=True)

    feed.on_update = show
    feed.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        feed.stop()


if __name__ == "__main__":
    main()
