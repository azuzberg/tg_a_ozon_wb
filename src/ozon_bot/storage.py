"""PostgreSQL storage for expiring metadata and imported data, without API keys."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime
from uuid import UUID

import asyncpg

from .imports import normalize_rows

_LOCK_ID = 0x4F5A4F4E424F5401
_SCHEMA = """
CREATE TABLE IF NOT EXISTS ozon_bot_sessions (
    session_id UUID PRIMARY KEY,
    user_id BIGINT NOT NULL UNIQUE,
    chat_id BIGINT NOT NULL,
    name TEXT NOT NULL CHECK (char_length(name) BETWEEN 1 AND 120),
    expires_at TIMESTAMPTZ NOT NULL,
    tax_rate NUMERIC(7,4) NOT NULL DEFAULT 6 CHECK (tax_rate BETWEEN 0 AND 100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ozon_bot_sessions_expires_idx
    ON ozon_bot_sessions (expires_at);
CREATE TABLE IF NOT EXISTS ozon_bot_imports (
    session_id UUID NOT NULL REFERENCES ozon_bot_sessions(session_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('costs', 'ads', 'seo', 'competitors')),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'array'),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, kind)
);
"""


class Storage:
    """One bot instance per database; credentials exist only in the caller's RAM.

    A dedicated PostgreSQL connection holds the advisory lock. On a successful
    restart all previous session metadata and imports are removed, since their
    in-memory API keys cannot be restored.
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None
        self._lock_connection: asyncpg.Connection | None = None
        self._lock_lost = asyncio.Event()
        self._closing = False

    @property
    def lock_lost(self) -> bool:
        return self._lock_lost.is_set()

    async def wait_lock_lost(self) -> None:
        """Wake a shutdown watchdog when the lock connection unexpectedly ends."""
        await self._lock_lost.wait()

    def _on_lock_terminated(self, connection: asyncpg.Connection) -> None:
        if not self._closing and self._lock_connection is connection:
            self._lock_lost.set()

    @property
    def pool(self) -> asyncpg.Pool:
        if (
            self._lock_connection is not None
            and self._lock_connection.is_closed()
            and not self._closing
        ):
            # A connection can report closed before its queued termination
            # callback runs. Refuse work during that interval as well.
            self._lock_lost.set()
        if self.lock_lost:
            raise RuntimeError("Соединение блокировки PostgreSQL потеряно. Перезапустите бота.")
        if self._pool is None:
            raise RuntimeError("Соединение с PostgreSQL ещё не открыто.")
        return self._pool

    async def open(self) -> None:
        if self._pool is not None:
            if self.lock_lost or self._lock_connection is None or self._lock_connection.is_closed():
                self._lock_lost.set()
                raise RuntimeError("Соединение блокировки PostgreSQL потеряно. Перезапустите бота.")
            return
        self._closing = False
        self._lock_lost.clear()
        connection = await asyncpg.connect(self._database_url, command_timeout=30)
        try:
            locked = await connection.fetchval("SELECT pg_try_advisory_lock($1::bigint)", _LOCK_ID)
            if not locked:
                raise RuntimeError("Другой экземпляр бота уже использует эту базу данных.")
            self._lock_connection = connection
            connection.add_termination_listener(self._on_lock_terminated)
            async with connection.transaction():
                await connection.execute(_SCHEMA)
                await connection.execute("DELETE FROM ozon_bot_sessions")
            pool = await asyncpg.create_pool(
                self._database_url, min_size=1, max_size=5, command_timeout=30
            )
            if connection.is_closed() or self.lock_lost:
                await pool.close()
                raise RuntimeError("Соединение блокировки PostgreSQL потеряно во время запуска.")
            self._pool = pool
        except BaseException:
            connection.remove_termination_listener(self._on_lock_terminated)
            if self._lock_connection is connection:
                self._lock_connection = None
            await connection.close()
            raise

    async def close(self) -> None:
        self._closing = True
        pool, self._pool = self._pool, None
        connection, self._lock_connection = self._lock_connection, None
        if connection is not None:
            connection.remove_termination_listener(self._on_lock_terminated)
        try:
            if pool is not None:
                await pool.close()
        finally:
            if connection is not None:
                # PostgreSQL releases all advisory locks when their owning
                # connection closes, including a connection already lost.
                await connection.close()

    async def create_session(
        self,
        session_id: str,
        user_id: int,
        chat_id: int,
        name: str,
        expires_at: datetime,
    ) -> None:
        _aware(expires_at)
        await self.pool.execute(
            """INSERT INTO ozon_bot_sessions
               (session_id, user_id, chat_id, name, expires_at)
               VALUES ($1, $2, $3, $4, $5)""",
            UUID(session_id),
            user_id,
            chat_id,
            name.strip(),
            expires_at,
        )

    async def delete_session(self, session_id: str) -> None:
        # Foreign-key cascade makes metadata and imports disappear atomically.
        await self.pool.execute(
            "DELETE FROM ozon_bot_sessions WHERE session_id = $1", UUID(session_id)
        )

    async def set_tax(self, session_id: str, rate: float) -> None:
        if isinstance(rate, bool) or not math.isfinite(rate) or not 0 <= rate <= 100:
            raise ValueError("Налоговая ставка должна быть от 0 до 100 процентов.")
        result = await self.pool.execute(
            "UPDATE ozon_bot_sessions SET tax_rate = $2 WHERE session_id = $1",
            UUID(session_id),
            rate,
        )
        if result == "UPDATE 0":
            raise ValueError("Сеанс уже завершён.")

    async def save_import(self, session_id: str, kind: str, rows: list[dict]) -> None:
        clean_rows = normalize_rows(kind, rows)
        payload = json.dumps(clean_rows, ensure_ascii=False, allow_nan=False)
        await self.pool.execute(
            """INSERT INTO ozon_bot_imports (session_id, kind, payload)
               VALUES ($1, $2, $3::jsonb)
               ON CONFLICT (session_id, kind)
               DO UPDATE SET payload = EXCLUDED.payload, updated_at = CURRENT_TIMESTAMP""",
            UUID(session_id),
            kind,
            payload,
        )

    async def load_imports(self, session_id: str) -> dict[str, list[dict]]:
        records = await self.pool.fetch(
            "SELECT kind, payload FROM ozon_bot_imports WHERE session_id = $1",
            UUID(session_id),
        )
        return {record["kind"]: json.loads(record["payload"]) for record in records}

    async def expire_sessions(self, now: datetime) -> list[str]:
        _aware(now)
        records = await self.pool.fetch(
            "DELETE FROM ozon_bot_sessions WHERE expires_at <= $1 RETURNING session_id", now
        )
        return [str(record["session_id"]) for record in records]

    async def touch(self, session_id: str, expires_at: datetime) -> None:
        _aware(expires_at)
        result = await self.pool.execute(
            "UPDATE ozon_bot_sessions SET expires_at = $2 WHERE session_id = $1",
            UUID(session_id),
            expires_at,
        )
        if result == "UPDATE 0":
            raise ValueError("Сеанс уже завершён.")


def _aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Время сеанса должно включать часовой пояс.")
