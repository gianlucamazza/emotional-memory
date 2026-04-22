"""In-memory affective state persistence for tests and short-lived apps."""

from __future__ import annotations

from emotional_memory.state import AffectiveState


class InMemoryAffectiveStateStore:
    """Simple process-local state store.

    The stored state is copied on save/load so callers cannot accidentally
    mutate the persisted snapshot by holding a shared reference.
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: AffectiveState | None = None

    def save(self, state: AffectiveState) -> None:
        self._state = state.model_copy()

    def load(self) -> AffectiveState | None:
        return None if self._state is None else self._state.model_copy()

    def clear(self) -> None:
        self._state = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(has_state={self._state is not None})"
