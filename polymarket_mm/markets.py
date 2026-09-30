"""Pick markets worth making: liquid, active, mid-priced, far from resolution."""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .config import Config

log = logging.getLogger(__name__)
GAMMA = "https://gamma-api.polymarket.com"


@dataclass(frozen=True)
class Market:
    question: str
    condition_id: str
    yes_token: str
    no_token: str
    mid_hint: float
    min_size: float
    tick: float
    rewards_max_spread: float | None  # in price units (0.02 = 2c), None if no rewards
    volume_24h: float


def _loads(v):
    return json.loads(v) if isinstance(v, str) else v


def parse_market(raw: dict, cfg: Config, now: datetime | None = None) -> Market | None:
    """Turn a Gamma market payload into a Market, or None if it isn't suitable."""
    now = now or datetime.now(timezone.utc)
    try:
        if raw.get("closed") or not raw.get("active"):
            return None
        if not raw.get("enableOrderBook") or not raw.get("acceptingOrders"):
            return None
        tokens = _loads(raw["clobTokenIds"])
        outcomes = _loads(raw["outcomes"])
        prices = [float(p) for p in _loads(raw["outcomePrices"])]
        if len(tokens) != 2 or [o.lower() for o in outcomes] != ["yes", "no"]:
            return None  # only plain Yes/No markets
        if not cfg.min_mid <= prices[0] <= cfg.max_mid:
            return None
        if float(raw.get("volume24hr") or 0) < cfg.min_volume_24h:
            return None
        end = raw.get("endDate")
        if end:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            if (end_dt - now).total_seconds() < cfg.min_days_to_end * 86400:
                return None
        rms = raw.get("rewardsMaxSpread")
        return Market(
            question=raw["question"],
            condition_id=raw["conditionId"],
            yes_token=tokens[0],
            no_token=tokens[1],
            mid_hint=prices[0],
            min_size=float(raw.get("orderMinSize") or 5),
            tick=float(raw.get("orderPriceMinTickSize") or 0.01),
            rewards_max_spread=float(rms) / 100 if rms else None,
            volume_24h=float(raw["volume24hr"]),
        )
    except (KeyError, ValueError, TypeError) as e:
        log.debug("skipping malformed market %s: %s", raw.get("slug"), e)
        return None


def select_markets(cfg: Config, session: requests.Session | None = None) -> list[Market]:
    """Top markets by 24h volume that pass the filters; reward-paying markets first."""
    s = session or requests.Session()
    resp = s.get(
        f"{GAMMA}/markets",
        params={
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": 100,
        },
        timeout=20,
    )
    resp.raise_for_status()
    found = [m for m in (parse_market(r, cfg) for r in resp.json()) if m]
    found.sort(key=lambda m: (m.rewards_max_spread is None, -m.volume_24h))
    return found[: cfg.max_markets]
