"""Async delegations participate in the bounded restart wait (regression for #148748)."""

import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner
from tests.waiting import async_wait_until
from tools import async_delegation


@pytest.mark.asyncio
async def test_restart_waits_for_delegation_after_parent_finishes(monkeypatch):
    runner, _ = make_restart_runner()
    runner.stop = AsyncMock()
    runner._restart_after_turn_timeout = 10.0
    probe = Mock(return_value=1)
    monkeypatch.setattr(async_delegation, "active_count", probe)
    runner._running_agents["parent"] = Mock()
    runner._scale_to_zero_status = Mock()
    runner.request_restart(via_service=True)
    try:
        await async_wait_until(lambda: runner._scale_to_zero_status.called)
        del runner._running_agents["parent"]
        reads = probe.call_count
        await async_wait_until(lambda: probe.call_count > reads or runner._restart_task.done())
        # The parent no longer holds the wait; the delegation must do so on its own.
        assert runner._active_work_count() == 1
        assert runner._awaitable_work_count() == 1
        runner.stop.assert_not_awaited()
        probe.return_value = 0
        await asyncio.wait_for(runner._restart_task, timeout=5)
        runner.stop.assert_awaited_once_with(
            restart=True, detached_restart=False, service_restart=True
        )
    finally:
        probe.return_value = 0
        await asyncio.wait_for(runner._restart_task, timeout=5)


@pytest.mark.asyncio
async def test_restart_delegation_cap_still_calls_stop(monkeypatch, caplog):
    runner, _ = make_restart_runner()
    runner.stop = AsyncMock()
    runner._restart_after_turn_timeout = 0.01
    monkeypatch.setattr(async_delegation, "active_count", lambda: 1)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner.request_restart(via_service=True)
        await asyncio.wait_for(runner._restart_task, timeout=5)
    assert "Restart after-turn wait timed out" in caplog.text
    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )
    assert runner._awaitable_work_count() == 1


@pytest.mark.asyncio
async def test_restart_delegation_probe_failure_does_not_block(monkeypatch):
    runner, _ = make_restart_runner()
    runner.stop = AsyncMock()
    probe = Mock(side_effect=RuntimeError("unreadable delegations"))
    monkeypatch.setattr(async_delegation, "active_count", probe)
    runner.request_restart(via_service=True)
    await asyncio.wait_for(runner._restart_task, timeout=5)
    probe.assert_called()
    assert runner._active_work_count() == runner._awaitable_work_count() == 0
    assert not runner._describe_active_work()
    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )


def test_restart_describes_delegations_without_observer_duplicates(monkeypatch):
    from gateway.run_active_work_verb import active_work_verb

    runner, _ = make_restart_runner()
    monkeypatch.setattr(async_delegation, "active_count", lambda: 1)
    assert runner._describe_active_work() == [{"kind": "delegation", "pid": os.getpid()}]
    monkeypatch.setattr(async_delegation, "list_async_delegations", lambda: [
        {"status": "running", "delegation_id": "deleg_test", "goal": "finish the build"}
    ])
    loop = SimpleNamespace(call_soon_threadsafe=lambda fn: fn())
    answer = active_work_verb(runner, loop)({})
    assert answer["active_agents"] == 1
    assert len(answer["active_work"]) == 1
    assert answer["active_work"][0]["delegation_id"] == "deleg_test"
