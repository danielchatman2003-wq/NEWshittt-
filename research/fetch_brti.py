"""Download historical BRTI (5Hz) from Kalshi's CF Benchmarks passthrough, one hour per request.

Keeps only the whole-second prints (settlement uses one print per second) as data/brti/YYYYMMDDHH.npy
with columns [unix_second, price]. Resumable and paced; backs off on 429/503.

  python research/fetch_brti.py --days 14
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polymarket_mm.brti import KalshiAuth  # noqa: E402

BASE = "https://external-api.kalshi.com"
PATH = "/trade-api/v2/cfbenchmarks/history/values"
OUT = Path(__file__).resolve().parent.parent / "data" / "brti"


def fetch_hour(auth: KalshiAuth, hour: datetime, session: requests.Session) -> np.ndarray:
    for attempt in range(8):
        r = session.get(BASE + PATH, params={"id": "BRTI", "timespan": "HOUR", "timestamp": hour.strftime("%Y-%m-%dT%H:%M:%SZ")},
                        headers=auth.headers("GET", PATH), timeout=60)
        if r.status_code in (429, 503):
            time.sleep(min(60, 2 ** attempt))
            continue
        r.raise_for_status()
        pay = r.json()["data"]["payload"]
        rows = [(t["time"] // 1000, float(t["value"])) for t in pay if t["time"] % 1000 == 0]
        return np.array(rows, dtype=np.float64)
    raise RuntimeError(f"gave up on {hour}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()
    load_dotenv()
    auth, s = KalshiAuth.from_env(), requests.Session()
    OUT.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)  # last complete hour is end-1h
    hours = [end - timedelta(hours=i) for i in range(1, int(args.days * 24) + 1)]  # newest first: most useful data lands first
    for i, h in enumerate(hours):
        f = OUT / f"{h:%Y%m%d%H}.npy"
        if f.exists():
            continue
        arr = fetch_hour(auth, h, s)
        np.save(f, arr)
        print(f"[{i + 1}/{len(hours)}] {h:%Y-%m-%d %H}Z  {len(arr)} prints", flush=True)
        time.sleep(args.pace)


if __name__ == "__main__":
    main()
