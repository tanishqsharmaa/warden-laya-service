import asyncio

import pytest

from warden_laya.concurrency import ConcurrencyGuard, ConcurrencyLimitExceededError


@pytest.mark.asyncio
async def test_concurrency_guard_permits_up_to_limit():
    guard = ConcurrencyGuard(max_concurrency=16)
    assert guard.active_slots == 0

    acquired = []
    async def worker():
        async with guard:
            acquired.append(True)
            await asyncio.sleep(0.05)

    tasks = [asyncio.create_task(worker()) for _ in range(16)]
    await asyncio.sleep(0.01)
    assert guard.active_slots == 16
    await asyncio.gather(*tasks)
    assert guard.active_slots == 0
    assert len(acquired) == 16

@pytest.mark.asyncio
async def test_concurrency_guard_sheds_slot_17_immediately():
    guard = ConcurrencyGuard(max_concurrency=16)

    async def holding_worker():
        async with guard:
            await asyncio.sleep(0.1)

    tasks = [asyncio.create_task(holding_worker()) for _ in range(16)]
    await asyncio.sleep(0.01)
    assert guard.active_slots == 16

    # 17th request must raise ConcurrencyLimitExceededError immediately (< 10ms)
    t0 = asyncio.get_event_loop().time()
    with pytest.raises(ConcurrencyLimitExceededError) as exc_info:
        async with guard:
            pass
    t_elapsed = asyncio.get_event_loop().time() - t0
    assert t_elapsed < 0.010  # Less than 10ms (sub-millisecond rejection)
    assert exc_info.value.retry_after == 0.1

    await asyncio.gather(*tasks)
    assert guard.active_slots == 0
