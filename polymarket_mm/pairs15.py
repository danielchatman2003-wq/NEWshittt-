"""Measure the cross-venue PAIR idea on the live 15-minute books (read-only, places no orders).

Buy Up on one venue and Down on the other: the pair always pays exactly $1, so it only makes money if
price + fees < $1.  Two combinations: A = Up@Kalshi + Down@Polymarket,  B = Up@Polymarket + Down@Kalshi.
Fees (taker): Kalshi KXBTC15M 0.07*p*(1-p), Polymarket US 0.0695*p*(1-p), per contract at each leg's price.

  python -m polymarket_mm.pairs15 [seconds]
"""
import logging
import sys
import time

from dotenv import load_dotenv

from .brti import BrtiFeed, KalshiAuth
from .hourly import find_current as find_pm
from .kalshi_feed import KalshiBookFeed
from .model import SecondSampler, fair_up
from .us_feed import UsAuth, UsBookFeed

K_FEE, PM_FEE = 0.07, 0.0695


def fee(coef: float, p: float) -> float:
    return coef * p * (1 - p)


def pair_costs(pm, ka):
    """Return {combo: (gross_cost, fees, net_cost)} for both combinations, or None if a book is empty."""
    if not pm or not ka or None in (pm.best_bid, pm.best_ask, ka.best_bid, ka.best_ask):
        return None
    out = {}
    up_k, dn_k = ka.best_ask, 1 - ka.best_bid
    up_p, dn_p = pm.best_ask, 1 - pm.best_bid
    for name, (up, up_fee, dn, dn_fee) in {
        "A Up@Kalshi+Down@PM": (up_k, fee(K_FEE, up_k), dn_p, fee(PM_FEE, dn_p)),
        "B Up@PM+Down@Kalshi": (up_p, fee(PM_FEE, up_p), dn_k, fee(K_FEE, dn_k)),
    }.items():
        gross, fees = up + dn, up_fee + dn_fee
        out[name] = (gross, fees, gross + fees)
    return out


def main() -> None:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 120
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    us, kal = UsAuth.from_env(), KalshiAuth.from_env()
    from .hourly_mm import seed_sampler
    sampler = SecondSampler(); seed_sampler(sampler, kal)
    brti = BrtiFeed(kal); brti.on_tick = lambda t: t.spot and sampler.add(t.local_ts, t.spot); brti.start()
    kfeed = KalshiBookFeed(kal); kfeed.start()
    pm_mkt, pm_feed = None, None
    rows, end = [], time.time() + secs
    try:
        while time.time() < end:
            now = time.time()
            if pm_mkt is None or now >= pm_mkt.window_end:
                nxt = find_pm(us, now, horizon="15m")
                if nxt is None:
                    time.sleep(2); continue
                if pm_feed: pm_feed.stop()
                pm_mkt, pm_feed = nxt, UsBookFeed(us, [nxt.slug]); pm_feed.start()
            pc = pair_costs(pm_feed.book(pm_mkt.slug), kfeed.book())
            if pc and sampler.last() and pm_mkt.price_to_beat:
                fair = fair_up(spot=sampler.last(), k=pm_mkt.price_to_beat, now=now, window_end=pm_mkt.window_end,
                               sigma=sampler.sigma(now), sampler=sampler)
                best = min(pc.items(), key=lambda kv: kv[1][2])
                rows.append((now, fair, best[0], best[1][0], best[1][1], best[1][2]))
                print(f"T-{pm_mkt.window_end - now:4.0f}s fair UP {fair:.3f} | best pair: {best[0]:<21} price {best[1][0]:.3f} + fees {best[1][1]:.3f} = {best[1][2]:.3f}"
                      f"  => {'PROFIT' if best[1][2] < 1 else 'loss'} {(1 - best[1][2]) * 100:+.1f}c per pair", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        kfeed.stop(); brti.stop()
        if pm_feed: pm_feed.stop()
    if rows:
        g = [r[3] for r in rows]; n = [r[5] for r in rows]
        print(f"\n{len(rows)} seconds observed.  COMBINED PRICE (before fees): min {min(g):.3f}  mean {sum(g) / len(g):.3f}   "
              f"AFTER FEES: min {min(n):.3f}  mean {sum(n) / len(n):.3f}")
        print(f"seconds where the pair costs < $1.00 before fees: {sum(x < 1 for x in g)}   after fees (real profit): {sum(x < 1 for x in n)}")
        print(f"best/typical profit per pair after fees: {(1 - min(n)) * 100:+.1f}c / {(1 - sum(n) / len(n)) * 100:+.1f}c")


if __name__ == "__main__":
    main()
