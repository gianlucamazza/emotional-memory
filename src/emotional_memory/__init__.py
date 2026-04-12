"""Emotional memory library for LLMs based on Affective Field Theory."""

import contextlib
from importlib.metadata import version

__version__ = version("emotional_memory")

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.appraisal import (
    AppraisalEngine,
    AppraisalVector,
    StaticAppraisalEngine,
    consolidation_strength,
)
from emotional_memory.appraisal_llm import (
    KeywordAppraisalEngine,
    KeywordRule,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
    LLMCallable,
)
from emotional_memory.async_adapters import (
    SyncToAsyncAppraisalEngine,
    SyncToAsyncEmbedder,
    SyncToAsyncStore,
    as_async,
)
from emotional_memory.async_engine import AsyncEmotionalMemory
from emotional_memory.decay import DecayConfig
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.interfaces import Embedder, MemoryStore, SequentialEmbedder
from emotional_memory.interfaces_async import AsyncAppraisalEngine, AsyncEmbedder, AsyncMemoryStore
from emotional_memory.models import EmotionalTag, Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodDecayConfig, MoodField
from emotional_memory.resonance import ResonanceConfig, hebbian_strengthen, spreading_activation
from emotional_memory.retrieval import AdaptiveWeightsConfig, RetrievalConfig
from emotional_memory.state import AffectiveState
from emotional_memory.stores.in_memory import InMemoryStore

with contextlib.suppress(ImportError):
    from emotional_memory.stores.sqlite import SQLiteStore as SQLiteStore

__all__ = [
    "AdaptiveWeightsConfig",
    "AffectiveMomentum",
    "AffectiveState",
    "AppraisalEngine",
    "AppraisalVector",
    "AsyncAppraisalEngine",
    "AsyncEmbedder",
    "AsyncEmotionalMemory",
    "AsyncMemoryStore",
    "CoreAffect",
    "DecayConfig",
    "Embedder",
    "EmotionalMemory",
    "EmotionalMemoryConfig",
    "EmotionalTag",
    "InMemoryStore",
    "KeywordAppraisalEngine",
    "KeywordRule",
    "LLMAppraisalConfig",
    "LLMAppraisalEngine",
    "LLMCallable",
    "Memory",
    "MemoryStore",
    "MoodDecayConfig",
    "MoodField",
    "ResonanceConfig",
    "ResonanceLink",
    "RetrievalConfig",
    "SQLiteStore",
    "SequentialEmbedder",
    "StaticAppraisalEngine",
    "SyncToAsyncAppraisalEngine",
    "SyncToAsyncEmbedder",
    "SyncToAsyncStore",
    "__version__",
    "as_async",
    "consolidation_strength",
    "hebbian_strengthen",
    "make_emotional_tag",
    "spreading_activation",
]
