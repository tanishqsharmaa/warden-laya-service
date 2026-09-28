import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConcurrencyLimitExceededError(Exception):
    """
    Raised when all in-flight inference slots are saturated.
    Maps to HTTP 429 Too Many Requests / gRPC StatusCode.RESOURCE_EXHAUSTED.
    """
    def __init__(
        self,
        message: str = "ERR_CONCURRENCY_LIMIT_EXCEEDED: Max concurrent inference slots saturated.",
        retry_after: float = 0.1,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class ConcurrencyGuard:
    """
    In-flight execution concurrency guard enforcing an upper ceiling (default 16 slots)
    with immediate non-blocking overload shedding.
    """
    def __init__(self, max_concurrency: int = 16) -> None:
        self._max_concurrency = max_concurrency
        self._active_slots = 0
        self._lock = asyncio.Lock()

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def active_slots(self) -> int:
        return self._active_slots

    async def __aenter__(self) -> "ConcurrencyGuard":
        async with self._lock:
            if self._active_slots >= self._max_concurrency:
                logger.warning(
                    f"Concurrency saturated ({self._active_slots}/{self._max_concurrency}). "
                    "Shedding incoming request immediately."
                )
                raise ConcurrencyLimitExceededError(retry_after=0.1)
            self._active_slots += 1
            return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        async with self._lock:
            self._active_slots = max(0, self._active_slots - 1)
