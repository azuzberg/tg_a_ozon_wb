"""Real SQLite tests, using temporary local files and no external services."""

import asyncio
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from ozon_bot.storage import Storage, StorageError


@pytest_asyncio.fixture
async def storage(tmp_path):
    instance = Storage(tmp_path / "nested" / "bot.sqlite3")
    await instance.open()
    try:
        yield instance
    finally:
        await instance.close()


def _query(storage, sql, parameters=()):
    with closing(sqlite3.connect(storage.database_path)) as connection:
        return connection.execute(sql, parameters).fetchall()


def _scalar(storage, sql, parameters=()):
    return _query(storage, sql, parameters)[0][0]


async def _session(storage, *, user_id=1001, expired=False, expires_at=None):
    session_id = str(uuid4())
    if expires_at is None:
        delta = timedelta(minutes=-1 if expired else 30)
        expires_at = datetime.now(UTC) + delta
    await storage.create_session(session_id, user_id, user_id, "Иван", expires_at)
    return session_id


@pytest.mark.asyncio
async def test_sqlite_file_and_parent_created_automatically_with_wal(storage):
    assert storage.database_path.is_file()
    assert _scalar(storage, "PRAGMA journal_mode") == "wal"
    async with storage._connection.execute("PRAGMA foreign_keys") as cursor:
        assert await cursor.fetchone() == (1,)
    async with storage._connection.execute("PRAGMA busy_timeout") as cursor:
        assert await cursor.fetchone() == (5000,)


@pytest.mark.asyncio
async def test_logout_cascades_all_imported_records(storage):
    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "00123", "cost": 35.5}])
    await storage.save_import(session_id, "seo", [{"sku": "00123", "title": "Товар"}])
    assert set(await storage.load_imports(session_id)) == {"costs", "seo"}
    await storage.delete_session(session_id)
    assert await storage.load_imports(session_id) == {}
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_sessions") == 0
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_imports") == 0
    # Repeated logout is harmless.
    await storage.delete_session(session_id)


@pytest.mark.asyncio
async def test_no_credentials_in_schema_and_unknown_fields_cannot_be_persisted(storage):
    expected_columns = {
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
    for table, allowed in expected_columns.items():
        columns = _query(storage, f"PRAGMA table_info({table})")
        assert {column[1] for column in columns} == allowed
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
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_imports") == 1


@pytest.mark.asyncio
async def test_import_without_active_session_cannot_be_saved(storage):
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        await storage.save_import(str(uuid4()), "costs", [{"sku": "001", "cost": 8}])
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_imports") == 0
    # A failed write must roll back so the next transaction can proceed.
    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "001", "cost": 8}])
    assert await storage.load_imports(session_id)


@pytest.mark.asyncio
async def test_expiry_cascades_only_expired_sessions(storage):
    expired = await _session(storage, expired=True)
    active = await _session(storage, user_id=1002)
    await storage.save_import(expired, "costs", [{"sku": "001", "cost": 5}])
    assert await storage.expire_sessions(datetime.now(UTC)) == [expired]
    assert await storage.load_imports(expired) == {}
    assert _scalar(storage, "SELECT session_id FROM ozon_bot_sessions") == active


@pytest.mark.asyncio
async def test_expiry_compares_instants_across_timezone_offsets(storage):
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    expired = await _session(
        storage,
        expires_at=now.astimezone(timezone(timedelta(hours=5))) - timedelta(minutes=1),
    )
    boundary = await _session(
        storage, user_id=1002, expires_at=now.astimezone(timezone(timedelta(hours=10)))
    )
    active = await _session(
        storage,
        user_id=1003,
        expires_at=now.astimezone(timezone(timedelta(hours=-7))) + timedelta(minutes=1),
    )
    assert set(await storage.expire_sessions(now)) == {expired, boundary}
    assert _scalar(storage, "SELECT session_id FROM ozon_bot_sessions") == active


@pytest.mark.asyncio
async def test_touch_extends_session_with_non_utc_timezone(storage):
    session_id = await _session(storage, expired=True)
    expires = (datetime.now(UTC) + timedelta(hours=1)).astimezone(timezone(timedelta(hours=-7)))
    await storage.touch(session_id, expires)
    assert await storage.expire_sessions(datetime.now(UTC)) == []


@pytest.mark.asyncio
async def test_naive_datetimes_rejected_without_changes(storage):
    naive = datetime(2026, 1, 1, 12)
    with pytest.raises(ValueError, match="часовой пояс"):
        await storage.create_session(str(uuid4()), 1001, 1001, "Иван", naive)
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_sessions") == 0
    session_id = await _session(storage)
    before = _scalar(storage, "SELECT expires_at FROM ozon_bot_sessions")
    with pytest.raises(ValueError, match="часовой пояс"):
        await storage.touch(session_id, naive)
    with pytest.raises(ValueError, match="часовой пояс"):
        await storage.expire_sessions(naive)
    assert _scalar(storage, "SELECT expires_at FROM ozon_bot_sessions") == before


@pytest.mark.asyncio
async def test_tax_bounds_enforced_in_python_and_sqlite(storage):
    session_id = await _session(storage)
    assert _scalar(storage, "SELECT tax_rate FROM ozon_bot_sessions") == 6
    for rate in (-1, 101, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            await storage.set_tax(session_id, rate)
    for rate in (0, 6, 15, 100):
        await storage.set_tax(session_id, rate)
        assert _scalar(storage, "SELECT tax_rate FROM ozon_bot_sessions") == rate
    with closing(sqlite3.connect(storage.database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute("UPDATE ozon_bot_sessions SET tax_rate = 101")
    assert _scalar(storage, "SELECT tax_rate FROM ozon_bot_sessions") == 100


@pytest.mark.asyncio
async def test_import_payload_schema_requires_valid_json_array(storage):
    session_id = await _session(storage)
    for payload in ("not JSON", '{"secret": "value"}', '"text"', "null"):
        with closing(sqlite3.connect(storage.database_path)) as connection:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(
                    "INSERT INTO ozon_bot_imports (session_id, kind, payload) VALUES (?, ?, ?)",
                    (session_id, "costs", payload),
                )
    assert await storage.load_imports(session_id) == {}


@pytest.mark.asyncio
async def test_second_instance_cannot_clear_live_sessions(storage):
    session_id = await _session(storage)
    other = Storage(storage.database_path)
    try:
        with pytest.raises(RuntimeError, match="Другой экземпляр"):
            await other.open()
        assert _scalar(storage, "SELECT session_id FROM ozon_bot_sessions") == session_id
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_second_process_cannot_clear_live_sessions(storage):
    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "001", "cost": 5}])
    child_code = """
import asyncio
import sys
from ozon_bot.storage import Storage

async def main():
    storage = Storage(sys.argv[1])
    try:
        await storage.open()
    except RuntimeError as exc:
        print(str(exc))
        return
    finally:
        await storage.close()
    raise AssertionError("Second process acquired the running bot's database")

asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        child_code,
        str(storage.database_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, stderr.decode()
    assert "Другой экземпляр" in stdout.decode()
    assert _scalar(storage, "SELECT session_id FROM ozon_bot_sessions") == session_id
    assert await storage.load_imports(session_id) == {"costs": [{"sku": "001", "cost": 5.0}]}


@pytest.mark.asyncio
async def test_restart_removes_sessions_whose_credentials_were_in_memory(storage):
    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "001", "cost": 5}])
    await storage.close()
    await storage.open()
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_sessions") == 0
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_imports") == 0
    # The same object remains usable after reopening.
    replacement = await _session(storage)
    await storage.set_tax(replacement, 15)
    assert _scalar(storage, "SELECT tax_rate FROM ozon_bot_sessions") == 15


@pytest.mark.asyncio
async def test_open_is_idempotent_without_purging_current_session(storage):
    session_id = await _session(storage)
    await storage.open()
    assert _scalar(storage, "SELECT session_id FROM ozon_bot_sessions") == session_id


@pytest.mark.asyncio
async def test_close_is_idempotent_and_closed_storage_rejects_work(storage):
    session_id = await _session(storage)
    await storage.close()
    await storage.close()
    with pytest.raises(RuntimeError, match="SQLite"):
        await storage.load_imports(session_id)
    with pytest.raises(RuntimeError, match="SQLite"):
        await storage.set_tax(session_id, 15)


@pytest.mark.asyncio
async def test_schema_failure_rolls_back_purge_and_releases_file_lock(storage, monkeypatch):
    import ozon_bot.storage as storage_module

    session_id = await _session(storage)
    await storage.save_import(session_id, "costs", [{"sku": "001", "cost": 5}])
    await storage.close()
    original_schema = storage_module._SCHEMA
    monkeypatch.setattr(
        storage_module,
        "_SCHEMA",
        original_schema
        + """
        CREATE TABLE setup_should_rollback (value TEXT);
        DELETE FROM ozon_bot_sessions;
        SELECT * FROM nonexistent_setup_table;
        """,
    )
    failing = Storage(storage.database_path)
    with pytest.raises(StorageError, match="Не удалось открыть SQLite") as error:
        await failing.open()
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
    assert _scalar(storage, "SELECT session_id FROM ozon_bot_sessions") == session_id
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_imports") == 1
    assert (
        _scalar(
            storage,
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'setup_should_rollback'",
        )
        == 0
    )
    monkeypatch.setattr(storage_module, "_SCHEMA", original_schema)
    # This fails if initialization kept either its transaction or file lock open.
    await storage.open()
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_sessions") == 0
    await failing.close()


@pytest.mark.asyncio
async def test_concurrent_writes_and_failed_transaction_stay_isolated(storage):
    original = await _session(storage)
    other = str(uuid4())
    results = await asyncio.gather(
        storage.create_session(str(uuid4()), 1001, 1001, "Duplicate", datetime.now(UTC)),
        storage.create_session(other, 1002, 1002, "Другой", datetime.now(UTC) + timedelta(hours=1)),
        storage.save_import(original, "costs", [{"sku": "001", "cost": 5}]),
        storage.set_tax(original, 15),
        return_exceptions=True,
    )
    assert isinstance(results[0], sqlite3.IntegrityError)
    assert results[1:] == [None, None, None]
    assert _query(storage, "SELECT session_id, name FROM ozon_bot_sessions ORDER BY user_id") == [
        (original, "Иван"),
        (other, "Другой"),
    ]
    assert (
        _scalar(storage, "SELECT tax_rate FROM ozon_bot_sessions WHERE session_id = ?", (original,))
        == 15
    )
    assert await storage.load_imports(original) == {"costs": [{"sku": "001", "cost": 5.0}]}


@pytest.mark.asyncio
async def test_repeated_cancellation_of_busy_begin_finishes_rollback(storage, monkeypatch):
    session_id = await _session(storage)
    begin_started = asyncio.Event()
    original_execute = storage._connection.execute

    def traced_execute(sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE":
            begin_started.set()
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(storage._connection, "execute", traced_execute)
    with closing(sqlite3.connect(storage.database_path)) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        task = asyncio.create_task(storage.set_tax(session_id, 15))
        try:
            await asyncio.wait_for(begin_started.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # Cancellation cannot free the operation lock while its queued
            # SQLite BEGIN is still blocked by this separate connection.
            assert not task.done()
        finally:
            blocker.rollback()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2)
    assert task.cancelled()
    assert _scalar(storage, "SELECT tax_rate FROM ozon_bot_sessions") == 6
    await storage.delete_session(session_id)
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_sessions") == 0


@pytest.mark.asyncio
async def test_finished_sessions_reject_tax_and_touch(storage):
    session_id = await _session(storage)
    await storage.delete_session(session_id)
    with pytest.raises(ValueError, match="Сеанс уже завершён"):
        await storage.set_tax(session_id, 15)
    with pytest.raises(ValueError, match="Сеанс уже завершён"):
        await storage.touch(session_id, datetime.now(UTC) + timedelta(hours=1))


@pytest.mark.asyncio
async def test_uuid_validation_and_canonical_storage(storage):
    session_id = str(uuid4())
    await storage.create_session(
        session_id.upper(), 1001, 1001, " Иван ", datetime.now(UTC) + timedelta(hours=1)
    )
    assert _query(storage, "SELECT session_id, name FROM ozon_bot_sessions") == [
        (session_id, "Иван")
    ]
    await storage.save_import(session_id.upper(), "costs", [{"sku": "001", "cost": 5}])
    assert await storage.load_imports(session_id)
    invalid = "not-a-uuid"
    for operation in (
        storage.create_session(invalid, 1002, 1002, "Иван", datetime.now(UTC)),
        storage.delete_session(invalid),
        storage.set_tax(invalid, 15),
        storage.save_import(invalid, "costs", [{"sku": "001", "cost": 5}]),
        storage.load_imports(invalid),
        storage.touch(invalid, datetime.now(UTC)),
    ):
        with pytest.raises(ValueError):
            await operation
    assert _scalar(storage, "SELECT COUNT(*) FROM ozon_bot_sessions") == 1
