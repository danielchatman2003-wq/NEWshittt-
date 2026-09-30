"""Fair value for a BTC Up/Down window that settles on averaged BRTI prints.

Settlement (Polymarket US): value(T) = mean of the 60 one-second BRTI prints in the 59s ending at T.
Up wins iff value(window_end) >= value(window_start)  (ties settle Up).  K = value(window_start) = priceToBeat.

We model BRTI as a driftless Brownian motion with per-sqrt-second dollar vol `sigma` (estimated
from recent BRTI), so the final 60s average is Normal with a closed-form mean/variance, even
part-way through the averaging window when some of its prints are already known.
"""
import math
from collections import deque

AVG_PRINTS = 60


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class SecondSampler:
    """One BRTI print per wall-clock second (last value seen in that second), with realised vol."""

    def __init__(self, keep: int = 1800):
        self.px: dict[int, float] = {}
        self._order: deque[int] = deque(maxlen=keep)

    def add(self, ts: float, price: float) -> None:
        s = int(ts)
        if s not in self.px:
            if len(self._order) == self._order.maxlen:
                self.px.pop(self._order[0], None)
            self._order.append(s)
        self.px[s] = price

    def last(self) -> float | None:
        return self.px[self._order[-1]] if self._order else None

    def window_avg(self, end_ts: float, n: int = AVG_PRINTS) -> float | None:
        """Mean of the n one-second prints ending at end_ts (inclusive). None if we lack most of them."""
        e = int(end_ts)
        vals = [self.px[s] for s in range(e - n + 1, e + 1) if s in self.px]
        return sum(vals) / len(vals) if len(vals) >= 0.8 * n else None

    def sigma(self, now: float, lookback: int = 900, lag: int = 10, floor: float = 2.0,
              prior: float = 6.0, prior_weight: int = 300) -> float:
        """Dollar vol per sqrt(second) from `lag`-second changes (less microstructure noise than 1s).

        Shrunk toward `prior` (~40% annual vol at $84k) until there's real history, so a short,
        noisy sample right after start-up can't swing fair value by several cents.
        """
        n = int(now)
        d = [self.px[s] - self.px[s - lag] for s in range(n - lookback, n + 1) if s in self.px and (s - lag) in self.px]
        if not d:
            return max(floor, prior)
        obs = math.sqrt(sum(x * x for x in d) / len(d) / lag)
        w = len(d) / (len(d) + prior_weight)
        return max(floor, w * obs + (1 - w) * prior)


def fair_up(*, spot: float, k: float, now: float, window_end: float, sigma: float,
            sampler: SecondSampler | None = None) -> float:
    """P(Up) = P(mean of final 60 prints >= K).

    tau = seconds to window_end. If tau > 60 the whole averaging window is in the future:
        F ~ N(spot, sigma^2 (tau - 60 + 20))     (gap of tau-60s, then a 60s time-average adds w/3)
    If tau <= 60, part of the window is already printed: F = (sum_known + m*avg_future)/60 with
    m = tau future prints -> mean (known + m*spot)/60, var (m/60)^2 sigma^2 m/3.
    """
    tau = window_end - now
    if tau <= 0:
        return 1.0 if spot >= k else 0.0
    if tau > AVG_PRINTS:
        mean, var = spot, sigma ** 2 * (tau - AVG_PRINTS + AVG_PRINTS / 3)
    else:
        m = int(math.ceil(tau))
        known = 0.0
        e = int(window_end)
        for s in range(e - AVG_PRINTS + 1, int(now) + 1):
            p = sampler.px.get(s) if sampler else None
            known += p if p is not None else spot
        mean = (known + m * spot) / AVG_PRINTS
        var = (m / AVG_PRINTS) ** 2 * sigma ** 2 * m / 3
    sd = math.sqrt(var)
    if sd < 0.5:  # essentially resolved; avoid a knife-edge on sub-dollar noise
        sd = 0.5
    return _phi((mean - k) / sd + 1e-9)
