"""Run against a disposable database selected by TEST_DATABASE_URL."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio

from ozon_bot.storage import Storage

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not configured")


@pytest_asyncio.fixture
async def storage():
    instance = Storage(DATABASE_URL)
    await instance.open()
    try:
        yield instance
    finally:
        await instance.close()


async def _session(storage, *, user_id=1001, expired=False):
    session_id = str(uuid4())
    delta = timedelta(minutes=-1 if expired else 30)
    await storage.create_session(session_id, user_id, user_id, "Иван", datetime.now(UTC) + delta)
    return session_id


@pytest.mark.asyncio
async def test_logout_cascades_all_imported_records(storage):
    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "00123", "cost": 35.5}])
    await storage.save_import(session_id, "seo", [{"sku": "00123", "title": "Товар"}])
    assert set(await storage.load_imports(session_id)) == {"costs", "seo"}
    await storage.delete_session(session_id)
    assert await storage.load_imports(session_id) == {}
    assert await storage.pool.fetchval("SELECT COUNT(*) FROM ozon_bot_sessions") == 0
    assert await storage.pool.fetchval("SELECT COUNT(*) FROM ozon_bot_imports") == 0


@pytest.mark.asyncio
async def test_no_credentials_in_schema_and_unknown_fields_cannot_be_persisted(storage):
    columns = await storage.pool.fetch(
        """SELECT table_name, column_name FROM information_schema.columns
           WHERE table_name IN ('ozon_bot_sessions', 'ozon_bot_imports')"""
    )
    allowed = {
        "ozon_bot_sessions": {
            "session_id",
            "user_id",
            "chat_id",
            "name",
            "expires_at",
            "tax_rate",
            "created_at",
        },
        "ozon_bot_imports": {"session_id", "kind", "payload", "updated_at"},
    }
    for column in columns:
        assert column["column_name"] in allowed[column["table_name"]]
    session_id = await _session(storage)
    with pytest.raises(ValueError):
        await storage.save_import(
            session_id, "costs", [{"sku": "001", "cost": 5, "api_key": "secret"}]
        )
    assert await storage.load_imports(session_id) == {}


@pytest.mark.asyncio
async def test_upsert_replaces_import_instead_of_appending(storage):
    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "001", "cost": 5}])
    await storage.save_import(session_id, "costs", [{"sku": "001", "cost": 8}])
    assert await storage.load_imports(session_id) == {"costs": [{"sku": "001", "cost": 8.0}]}


@pytest.mark.asyncio
async def test_expiry_cascades_only_expired_sessions(storage):
    expired = await _session(storage, expired=True)
    active = await _session(storage, user_id=1002)
    await storage.save_import(expired, "costs", [{"sku": "001", "cost": 5}])
    assert await storage.expire_sessions(datetime.now(UTC)) == [expired]
    assert await storage.load_imports(expired) == {}
    remaining = await storage.pool.fetchval("SELECT session_id FROM ozon_bot_sessions")
    assert str(remaining) == active


@pytest.mark.asyncio
async def test_touch_extends_session(storage):
    session_id = await _session(storage, expired=True)
    expires = datetime.now(UTC) + timedelta(hours=1)
    await storage.touch(session_id, expires)
    assert await storage.expire_sessions(datetime.now(UTC)) == []


@pytest.mark.asyncio
async def test_tax_bounds_enforced_in_python_and_postgresql(storage):
    session_id = await _session(storage)
    assert await storage.pool.fetchval("SELECT tax_rate FROM ozon_bot_sessions") == 6
    for rate in (-1, 101, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            await storage.set_tax(session_id, rate)
    for rate in (0, 6, 15, 100):
        await storage.set_tax(session_id, rate)
    with pytest.raises(asyncpg.CheckViolationError):
        await storage.pool.execute("UPDATE ozon_bot_sessions SET tax_rate = 101")


@pytest.mark.asyncio
async def test_second_process_cannot_clear_live_sessions(storage):
    session_id = await _session(storage)
    other = Storage(DATABASE_URL)
    with pytest.raises(RuntimeError, match="Другой экземпляр"):
        await other.open()
    assert (
        str(await storage.pool.fetchval("SELECT session_id FROM ozon_bot_sessions")) == session_id
    )
    await other.close()


@pytest.mark.asyncio
async def test_restart_removes_sessions_whose_credentials_were_in_memory(storage):
    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "001", "cost": 5}])
    await storage.close()
    await storage.open()
    assert await storage.pool.fetchval("SELECT COUNT(*) FROM ozon_bot_sessions") == 0
    assert await storage.pool.fetchval("SELECT COUNT(*) FROM ozon_bot_imports") == 0


@pytest.mark.asyncio
async def test_schema_failure_rolls_back_and_releases_advisory_lock(storage, monkeypatch):
    import ozon_bot.storage as storage_module

    session_id = await _session(storage)
    await storage.close()
    original_schema = storage_module._SCHEMA
    monkeypatch.setattr(storage_module, "_SCHEMA", original_schema + "SELECT 1 / 0;")
    failing = Storage(DATABASE_URL)
    with pytest.raises(asyncpg.DivisionByZeroError):
        await failing.open()
    connection = await asyncpg.connect(DATABASE_URL)
    try:
        assert (
            str(await connection.fetchval("SELECT session_id FROM ozon_bot_sessions")) == session_id
        )
    finally:
        await connection.close()
    monkeypatch.setattr(storage_module, "_SCHEMA", original_schema)
    # This would fail if the initialization error left the lock connection open.
    await storage.open()
    assert await storage.pool.fetchval("SELECT COUNT(*) FROM ozon_bot_sessions") == 0


@pytest.mark.asyncio
async def test_unexpected_lock_disconnect_wakes_watchdog_and_rejects_old_owner(storage):
    session_id = await _session(storage)
    watchdog = asyncio.create_task(storage.wait_lock_lost())
    assert not storage.lock_lost
    pid = await storage._lock_connection.fetchval("SELECT pg_backend_pid()")
    connection = await asyncpg.connect(DATABASE_URL)
    try:
        assert await connection.fetchval("SELECT pg_terminate_backend($1)", pid)
    finally:
        await connection.close()
    await asyncio.wait_for(watchdog, timeout=3)
    assert storage.lock_lost
    with pytest.raises(RuntimeError, match="блокировки"):
        await storage.load_imports(session_id)
    with pytest.raises(RuntimeError, match="блокировки"):
        await storage.touch(session_id, datetime.now(UTC) + timedelta(minutes=30))
    with pytest.raises(RuntimeError, match="блокировки"):
        await storage.open()
    # The server released the old lock, so a new instance can own the database;
    # the disconnected first instance must remain unable to access its pool.
    replacement = Storage(DATABASE_URL)
    await replacement.open()
    try:
        replacement_session = await _session(replacement, user_id=1002)
        await replacement.save_import(replacement_session, "costs", [{"sku": "001", "cost": 5}])
        with pytest.raises(RuntimeError, match="блокировки"):
            await storage.save_import(session_id, "costs", [{"sku": "002", "cost": 8}])
        assert await replacement.load_imports(replacement_session) == {
            "costs": [{"sku": "001", "cost": 5.0}]
        }
    finally:
        await replacement.close()
    # Shutdown after a lost connection is safe and does not issue SQL on it.
    await storage.close()


@pytest.mark.asyncio
async def test_intentional_close_does_not_signal_lost_lock(storage):
    watchdog = asyncio.create_task(storage.wait_lock_lost())
    await storage.close()
    await asyncio.sleep(0)
    assert not storage.lock_lost
    assert not watchdog.done()
    watchdog.cancel()
    await asyncio.gather(watchdog, return_exceptions=True)
