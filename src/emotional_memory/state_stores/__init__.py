import contextlib

from emotional_memory.state_stores.in_memory import InMemoryAffectiveStateStore
from emotional_memory.state_stores.redis import RedisAffectiveStateStore

__all__ = [
    "InMemoryAffectiveStateStore",
    "RedisAffectiveStateStore",
    "SQLiteAffectiveStateStore",
]

with contextlib.suppress(ImportError):
    from emotional_memory.state_stores.sqlite import (
        SQLiteAffectiveStateStore as SQLiteAffectiveStateStore,
    )
