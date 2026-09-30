import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Config:
    max_markets: int = 3
    quote_size: float = 20.0
    half_spread: float = 0.01
    skew: float = 0.02
    max_position: float = 100.0
    max_open_notional: float = 150.0
    min_volume_24h: float = 20_000.0
    min_mid: float = 0.15
    max_mid: float = 0.85
    min_days_to_end: float = 3.0
    max_book_spread: float = 0.06
    move_pause: float = 0.03
    refresh_seconds: float = 15.0
    max_errors: int = 5

    @classmethod
    def from_env(cls) -> "Config":
        """Read MM_<FIELD> overrides from the environment."""
        kwargs = {}
        for f in fields(cls):
            raw = os.getenv(f"MM_{f.name.upper()}")
            if raw not in (None, ""):
                kwargs[f.name] = type(f.default)(raw)
        return cls(**kwargs)


@dataclass(frozen=True)
class Credentials:
    private_key: str
    funder: str
    signature_type: int = 1

    @classmethod
    def from_env(cls) -> "Credentials":
        key = os.getenv("PRIVATE_KEY", "").strip()
        funder = os.getenv("FUNDER_ADDRESS", "").strip()
        if not key or not funder:
            raise SystemExit("--live requires PRIVATE_KEY and FUNDER_ADDRESS in the environment/.env")
        return cls(key, funder, int(os.getenv("SIGNATURE_TYPE", "1")))
