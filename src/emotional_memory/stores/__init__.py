import contextlib

from emotional_memory.stores.in_memory import InMemoryStore

__all__ = ["InMemoryStore"]

with contextlib.suppress(ImportError):
    from emotional_memory.stores.sqlite import SQLiteStore as SQLiteStore
