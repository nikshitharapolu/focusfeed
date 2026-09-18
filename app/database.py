from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://focusfeed:focusfeed@localhost:5432/focusfeed",
)


@contextmanager
def connection() -> Iterator[psycopg.Connection[Any]]:
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as database:
        yield database


def initialize_database() -> None:
    with connection() as database:
        database.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email VARCHAR(254) NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        database.execute("""
            CREATE TABLE IF NOT EXISTS user_state (
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                state_key VARCHAR(32) NOT NULL,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (user_id, state_key)
            )
        """)


def read_state(user_id: str, key: str, default: Any = None) -> Any:
    with connection() as database:
        row = database.execute(
            "SELECT payload FROM user_state WHERE user_id = %s AND state_key = %s",
            (user_id, key),
        ).fetchone()
    return row["payload"] if row else default


def write_state(user_id: str, key: str, payload: Any) -> None:
    with connection() as database:
        database.execute("""
            INSERT INTO user_state (user_id, state_key, payload)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id, state_key) DO UPDATE
            SET payload = EXCLUDED.payload, updated_at = NOW()
        """, (user_id, key, Jsonb(payload)))


def list_user_ids_with_profiles() -> list[str]:
    with connection() as database:
        rows = database.execute(
            "SELECT user_id::text AS id FROM user_state WHERE state_key = 'profile'"
        ).fetchall()
    return [row["id"] for row in rows]
