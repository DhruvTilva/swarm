from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, List, Optional

import aiosqlite


@dataclass
class SwarmMessage:
    source: str
    text: str
    target: str = "all"
    phase: str = "SYSTEM"
    kind: str = "chat"
    status: str = "thinking"
    timestamp: float = field(default_factory=time.time)


Subscriber = Callable[[SwarmMessage], Awaitable[None]]


class MessageBus:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._queue: Optional[asyncio.Queue[SwarmMessage]] = None
        self._subscribers: List[Subscriber] = []
        self._db: Optional[aiosqlite.Connection] = None
        self._lock: Optional[asyncio.Lock] = None
        self._message_count = 0

    def get_queue(self) -> asyncio.Queue[SwarmMessage]:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    @property
    def message_count(self) -> int:
        return self._message_count

    async def initialize(self) -> None:
        self._lock = asyncio.Lock()
        self._db = await aiosqlite.connect(self.database_path.as_posix())
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                phase TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS phases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                progress INTEGER NOT NULL,
                total INTEGER NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        await self._db.commit()

    async def subscribe(self, callback: Subscriber) -> None:
        self._subscribers.append(callback)

    async def publish(self, message: SwarmMessage) -> None:
        await self.get_queue().put(message)
        self._message_count += 1
        await self._persist_message(message)
        for subscriber in self._subscribers:
            try:
                await subscriber(message)
            except Exception:
                # A subscriber failure should not interrupt the swarm.
                continue

    async def set_phase(self, phase: str, progress: int, total: int) -> None:
        if self._lock is None:
            return
        async with self._lock:
            if self._db is None:
                return
            await self._db.execute(
                "INSERT INTO phases (phase, progress, total, timestamp) VALUES (?, ?, ?, ?)",
                (phase, progress, total, time.time()),
            )
            await self._db.commit()

    async def stream(self) -> AsyncIterator[SwarmMessage]:
        while True:
            item = await self.get_queue().get()
            yield item

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _persist_message(self, message: SwarmMessage) -> None:
        if self._lock is None:
            return
        async with self._lock:
            if self._db is None:
                return
            await self._db.execute(
                """
                INSERT INTO messages (source, target, phase, kind, status, text, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.source,
                    message.target,
                    message.phase,
                    message.kind,
                    message.status,
                    message.text,
                    message.timestamp,
                ),
            )
            await self._db.commit()
