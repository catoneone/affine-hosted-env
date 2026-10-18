"""Gateway rate limits (architecture v2): token buckets per source IP, per hotkey and for the whole gateway.

Rate limits answer "how fast"; the coordinator's quotas answer "how much". Over the limit the gateway returns
429 `rate_limited` with `Retry-After`; clients wait and resend (data-plane resends are idempotent).

Buckets live in process memory. `RateLimitBackend` is the seam for a shared store once the gateway runs as
several replicas.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Rate:
    per_sec: float      # sustained refill rate
    burst: float        # bucket size

    def __post_init__(self):
        if self.per_sec <= 0 or self.burst < 1:
            raise ValueError(f'bad rate {self}')


@dataclass(frozen=True)
class RateLimits:
    """None disables a limit. Defaults are generous on purpose: they stop floods, not normal mining
    (9 tasks x 10 concurrent sessions per hotkey, each long-polling `turn`)."""
    ip: Rate | None = Rate(500, 1000)               # per source IP, checked before authentication
    hotkey_control: Rate | None = Rate(5, 20)       # per hotkey: tasks/open, sessions/open, tasks/finish
    hotkey_data: Rate | None = Rate(200, 400)       # per hotkey: start / turn
    total: Rate | None = Rate(5000, 10000)          # whole gateway

    @classmethod
    def from_config(cls, cfg: dict | None) -> RateLimits:
        """`{scope: [per_sec, burst] | false}`; scopes left out keep their defaults."""
        kw = {}
        for scope, v in (cfg or {}).items():
            if scope not in cls.__dataclass_fields__:
                raise ValueError(f'unknown rate limit scope {scope!r}')
            kw[scope] = None if v in (False, None) else Rate(float(v[0]), float(v[1]))
        return cls(**kw)


class RateLimitBackend(Protocol):
    def check(self, scope: str, key: str, rate: Rate, cost: float = 1.0) -> float:
        """Take `cost` tokens. Returns 0 when allowed, otherwise the seconds until enough tokens refill."""


class MemoryBuckets:
    """Thread-safe in-memory token buckets. Keeps at most `max_keys` buckets; the least recently used are
    dropped first (a dropped bucket comes back full, which only ever errs on the side of allowing)."""

    def __init__(self, max_keys: int = 100_000, clock: Callable[[], float] = time.monotonic):
        self.max_keys, self.clock = max_keys, clock
        self._b: OrderedDict[tuple[str, str], tuple[float, float]] = OrderedDict()   # -> (tokens, last)
        self._mu = threading.Lock()

    def check(self, scope: str, key: str, rate: Rate, cost: float = 1.0) -> float:
        now = self.clock()
        k = (scope, key)
        with self._mu:
            tokens, last = self._b.pop(k, (rate.burst, now))
            tokens = min(rate.burst, tokens + (now - last) * rate.per_sec)
            if tokens >= cost:
                tokens, wait = tokens - cost, 0.0
            else:
                wait = (cost - tokens) / rate.per_sec
            self._b[k] = (tokens, now)
            while len(self._b) > self.max_keys:
                self._b.popitem(last=False)
        return wait


class RateLimitError(Exception):
    status = 429

    def __init__(self, scope: str, wait: float):
        super().__init__(f'rate_limited: {scope}')
        self.code, self.detail = 'rate_limited', scope
        self.retry_after = max(1, math.ceil(wait))


class RateLimiter:
    def __init__(self, limits: RateLimits = RateLimits(), backend: RateLimitBackend | None = None):
        self.limits, self.backend = limits, backend or MemoryBuckets()

    def _take(self, scope: str, key: str) -> None:
        rate = getattr(self.limits, scope)
        if rate is None:
            return
        wait = self.backend.check(scope, key, rate)
        if wait > 0:
            raise RateLimitError(scope, wait)

    def before_auth(self, ip: str) -> None:
        """Every request: the whole-gateway bucket, then the source IP's bucket."""
        self._take('total', '*')
        self._take('ip', ip or '?')

    def control(self, hotkey: str) -> None:
        self._take('hotkey_control', hotkey)

    def data(self, hotkey: str) -> None:
        self._take('hotkey_data', hotkey)
