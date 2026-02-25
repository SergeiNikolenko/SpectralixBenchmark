from __future__ import annotations

from typing import Callable, Iterable, TypeVar
import time

T = TypeVar("T")


def run_with_retries(
    operation: Callable[[], T],
    *,
    retries: int,
    retry_on: tuple[type[BaseException], ...],
    backoff_sec: float = 0.0,
) -> T:
    max_retries = max(0, int(retries))
    last_error: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except retry_on as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            if backoff_sec > 0:
                time.sleep(backoff_sec * (attempt + 1))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unreachable retry state")
