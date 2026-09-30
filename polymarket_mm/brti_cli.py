"""Print the live BRTI feed:  python -m polymarket_mm.brti_cli [-v]"""
import argparse
import logging
import time

from dotenv import load_dotenv

from .brti import BrtiFeed, KalshiAuth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="log raw non-tick frames (use to inspect the payload)")
    args = ap.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)

    feed = BrtiFeed(KalshiAuth.from_env())
    feed.on_tick = lambda t: print(
        f"BRTI spot={t.spot}  avg60={t.avg_60s} ({t.avg_60s_ticks}/60)  15m_avg={t.quarter_hour_avg}  "
        f"lag={t.local_ts - t.received_at:.2f}s",
        flush=True,
    )
    feed.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        feed.stop()


if __name__ == "__main__":
    main()
