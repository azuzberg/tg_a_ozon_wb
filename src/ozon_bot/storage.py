"""Local SQLite storage for expiring metadata and imports, without API keys."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import UUID

import aiosqlite
from filelock import FileLock, Timeout

from .imports import normalize_rows

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ozon_bot_sessions (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    chat_id INTEGER NOT NULL,
    name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 120),
    expires_at REAL NOT NULL,
    tax_rate REAL NOT NULL DEFAULT 6 CHECK (tax_rate BETWEEN 0 AND 100),
    created_at REAL NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS REAL))
);
CREATE INDEX IF NOT EXISTS ozon_bot_sessions_expires_idx
    ON ozon_bot_sessions (expires_at);
CREATE TABLE IF NOT EXISTS ozon_bot_imports (
    session_id TEXT NOT NULL REFERENCES ozon_bot_sessions(session_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('costs', 'ads', 'seo', 'competitors')),
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload) = 'array'),
    updated_at REAL NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS REAL)),
    PRIMARY KEY (session_id, kind)
);
"""


class StorageError(RuntimeError):
    """A startup or ownership error whose message is safe to show to the user."""


class Storage:
    """One bot instance per SQLite file; API credentials stay in the caller's RAM.

    A portable OS file lock is held for the lifetime of the connection. Only its
    owner may clear old sessions on startup, because their in-memory keys cannot
    be restored. All operations share an asyncio lock to keep transactions apart.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self._file_lock = FileLock(f"{self.database_path}.lock", thread_local=False)
        self._operation_lock = asyncio.Lock()
        self._connection: aiosqlite.Connection | None = None

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise StorageError("Соединение с SQLite ещё не открыто.")
        return self._connection

    async def open(self) -> None:
        async with self._operation_lock:
            if self._connection is not None:
                return
            try:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                self._file_lock.acquire(timeout=0)
            except Timeout as exc:
                raise StorageError("Другой экземпляр бота уже использует эту базу данных.") from exc
            except OSError as exc:
                raise StorageError(
                    "Не удалось создать файл SQLite. Проверьте DB_PATH и права на папку."
                ) from exc
            connection: aiosqlite.Connection | None = None
            try:
                connection = await aiosqlite.connect(self.database_path, isolation_level=None)
                for pragma in (
                    "PRAGMA foreign_keys = ON",
                    "PRAGMA busy_timeout = 5000",
                    "PRAGMA journal_mode = WAL",
                ):
                    async with connection.execute(pragma):
                        pass
                async with _transaction(connection):
                    # executescript implicitly commits an existing transaction.
                    # Execute the fixed schema statements individually instead.
                    for statement in _SCHEMA.split(";"):
                        if statement.strip():
                            async with connection.execute(statement):
                                pass
                    async with connection.execute("DELETE FROM ozon_bot_sessions"):
                        pass
                self._connection = connection
            except BaseException as exc:
                try:
                    if connection is not None:
                        await connection.close()
                finally:
                    self._file_lock.release()
                if isinstance(exc, (sqlite3.Error, OSError)):
                    raise StorageError(
                        "Не удалось открыть SQLite. Проверьте DB_PATH и права на папку."
                    ) from exc
                raise

    async def close(self) -> None:
        async with self._operation_lock:
            connection, self._connection = self._connection, None
            try:
                if connection is not None:
                    await connection.close()
            finally:
                self._file_lock.release()

    async def _write(self, sql: str, parameters: tuple, *, require_session: bool = False) -> None:
        async with self._operation_lock:
            connection = self._require_connection()
            async with _transaction(connection):
                async with connection.execute(sql, parameters) as cursor:
                    if require_session and cursor.rowcount == 0:
                        raise ValueError("Сеанс уже завершён.")

    async def create_session(
        self,
        session_id: str,
        user_id: int,
        chat_id: int,
        name: str,
        expires_at: datetime,
    ) -> None:
        await self._write(
            """INSERT INTO ozon_bot_sessions
               (session_id, user_id, chat_id, name, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            (str(UUID(session_id)), user_id, chat_id, name.strip(), _timestamp(expires_at)),
        )

    async def delete_session(self, session_id: str) -> None:
        # Foreign-key cascade removes metadata and imports in one transaction.
        await self._write(
            "DELETE FROM ozon_bot_sessions WHERE session_id = ?", (str(UUID(session_id)),)
        )

    async def set_tax(self, session_id: str, rate: float) -> None:
        if isinstance(rate, bool) or not math.isfinite(rate) or not 0 <= rate <= 100:
            raise ValueError("Налоговая ставка должна быть от 0 до 100 процентов.")
        await self._write(
            "UPDATE ozon_bot_sessions SET tax_rate = ? WHERE session_id = ?",
            (rate, str(UUID(session_id))),
            require_session=True,
        )

    async def save_import(self, session_id: str, kind: str, rows: list[dict]) -> None:
        clean_rows = normalize_rows(kind, rows)
        payload = json.dumps(clean_rows, ensure_ascii=False, allow_nan=False)
        await self._write(
            """INSERT INTO ozon_bot_imports (session_id, kind, payload)
               VALUES (?, ?, ?)
               ON CONFLICT (session_id, kind)
               DO UPDATE SET payload = excluded.payload,
                   updated_at = CAST(strftime('%s', 'now') AS REAL)""",
            (str(UUID(session_id)), kind, payload),
        )

    async def load_imports(self, session_id: str) -> dict[str, list[dict]]:
        canonical_id = str(UUID(session_id))
        async with self._operation_lock:
            connection = self._require_connection()
            async with connection.execute(
                "SELECT kind, payload FROM ozon_bot_imports WHERE session_id = ?",
                (canonical_id,),
            ) as cursor:
                records = await cursor.fetchall()
        return {kind: json.loads(payload) for kind, payload in records}

    async def expire_sessions(self, now: datetime) -> list[str]:
        timestamp = _timestamp(now)
        async with self._operation_lock:
            connection = self._require_connection()
            async with _transaction(connection):
                async with connection.execute(
                    "SELECT session_id FROM ozon_bot_sessions WHERE expires_at <= ?",
                    (timestamp,),
                ) as cursor:
                    records = await cursor.fetchall()
                async with connection.execute(
                    "DELETE FROM ozon_bot_sessions WHERE expires_at <= ?", (timestamp,)
                ):
                    pass
        return [session_id for (session_id,) in records]

    async def touch(self, session_id: str, expires_at: datetime) -> None:
        await self._write(
            "UPDATE ozon_bot_sessions SET expires_at = ? WHERE session_id = ?",
            (_timestamp(expires_at), str(UUID(session_id))),
            require_session=True,
        )


@asynccontextmanager
async def _transaction(connection: aiosqlite.Connection) -> AsyncIterator[None]:
    try:
        async with connection.execute("BEGIN IMMEDIATE"):
            pass
        yield
        await connection.commit()
    except BaseException:
        # aiosqlite queues SQL on a worker thread. Cancelling a waiting BEGIN
        # does not remove it from that queue: rollback must run afterwards and
        # finish before another coroutine can acquire the operation lock.
        rollback = asyncio.create_task(connection.rollback())
        while True:
            try:
                await asyncio.shield(rollback)
                break
            except asyncio.CancelledError:
                if rollback.cancelled():
                    raise
        raise


def _timestamp(value: datetime) -> float:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Время сеанса должно включать часовой пояс.")
    # Numeric UTC epochs compare instants correctly regardless of offset.
    return value.timestamp()
