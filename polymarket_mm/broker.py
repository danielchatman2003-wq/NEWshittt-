"""Order placement/inventory behind one interface: LiveBroker trades, DryRunBroker only logs."""
import itertools
import logging

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    OpenOrderParams,
    OrderArgs,
    OrderType,
)

from .config import Credentials
from .quoter import Quote

log = logging.getLogger(__name__)
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon


def make_public_client() -> ClobClient:
    return ClobClient(HOST)


def make_live_client(creds: Credentials) -> ClobClient:
    client = ClobClient(
        HOST,
        key=creds.private_key,
        chain_id=CHAIN_ID,
        signature_type=creds.signature_type,
        funder=creds.funder,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


def _open_order(o: dict) -> dict:
    remaining = float(o["original_size"]) - float(o.get("size_matched") or 0)
    return {
        "id": o["id"],
        "asset_id": o["asset_id"],
        "side": o["side"],
        "price": float(o["price"]),
        "size": remaining,
    }


class LiveBroker:
    def __init__(self, client: ClobClient):
        self.client = client

    def collateral_balance(self) -> float:
        r = self.client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        return float(r["balance"]) / 1e6

    def position(self, token_id: str) -> float:
        r = self.client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
        )
        return float(r["balance"]) / 1e6

    def open_orders(self, condition_id: str) -> list[dict]:
        return [_open_order(o) for o in self.client.get_orders(OpenOrderParams(market=condition_id))]

    def place(self, q: Quote) -> str | None:
        """Post a maker-only GTC order. Returns the order id, or None if it was rejected."""
        try:
            order = self.client.create_order(
                OrderArgs(token_id=q.token_id, price=q.price, size=round(q.size, 2), side=q.side)
            )
            resp = self.client.post_order(order, OrderType.GTC, post_only=True)
        except Exception as e:  # a rejected quote (e.g. would cross) must not kill the loop
            log.warning("place failed %s: %s", q, e)
            return None
        if not resp.get("success"):
            log.warning("place rejected %s: %s", q, resp.get("errorMsg"))
            return None
        return resp.get("orderID")

    def cancel(self, order_ids: list[str]) -> None:
        if order_ids:
            self.client.cancel_orders(order_ids)

    def cancel_market(self, condition_id: str) -> None:
        self.client.cancel_market_orders(market=condition_id)


class DryRunBroker:
    """Tracks would-be orders in memory so the reconcile logic runs; no fills are simulated."""

    def __init__(self):
        self._orders: dict[str, dict] = {}
        self._market_of: dict[str, str] = {}
        self._ids = itertools.count(1)

    def position(self, token_id: str) -> float:
        return 0.0

    def open_orders(self, condition_id: str) -> list[dict]:
        return [o for i, o in self._orders.items() if self._market_of[i] == condition_id]

    def place(self, q: Quote, condition_id: str = "") -> str:
        oid = f"dry-{next(self._ids)}"
        self._orders[oid] = {"id": oid, "asset_id": q.token_id, "side": q.side, "price": q.price, "size": q.size}
        self._market_of[oid] = condition_id
        log.info("[dry-run] PLACE %s %.2f @ %.3f (%s...)", q.side, q.size, q.price, q.token_id[:8])
        return oid

    def cancel(self, order_ids: list[str]) -> None:
        for oid in order_ids:
            if self._orders.pop(oid, None):
                self._market_of.pop(oid, None)
                log.info("[dry-run] CANCEL %s", oid)

    def cancel_market(self, condition_id: str) -> None:
        self.cancel([i for i, c in self._market_of.items() if c == condition_id])
