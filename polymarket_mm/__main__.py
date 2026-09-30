import argparse
import logging
import signal
import sys

from dotenv import load_dotenv

from .bot import MarketMaker
from .broker import DryRunBroker, LiveBroker, make_live_client, make_public_client
from .config import Config, Credentials


def main() -> int:
    ap = argparse.ArgumentParser(description="Polymarket market maker (dry-run unless --live)")
    ap.add_argument("--live", action="store_true", help="place REAL orders with real funds")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("polymarket_mm")
    cfg = Config.from_env()

    if args.live:
        clob = make_live_client(Credentials.from_env())
        broker = LiveBroker(clob)
        log.warning("LIVE MODE - collateral balance: $%.2f", broker.collateral_balance())
    else:
        clob, broker = make_public_client(), DryRunBroker()
        log.info("DRY-RUN: reading real books, no orders will be placed (pass --live to trade)")

    bot = MarketMaker(cfg, clob, broker)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: bot.stop.set())
    return bot.run()


if __name__ == "__main__":
    sys.exit(main())
