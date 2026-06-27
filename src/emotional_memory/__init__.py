"""Emotional memory library for LLMs based on Affective Field Theory."""

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

__version__ = version("emotional_memory")

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.appraisal import (
    AppraisalEngine,
    AppraisalVector,
    GenericAppraisalVector,
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
from emotional_memory.appraisal_schema import (
    DIRECT_VAD_SCHEMA,
    SCHERER_CPM_SCHEMA,
    AppraisalDimension,
    AppraisalSchema,
)
from emotional_memory.async_adapters import (
    SyncToAsyncAppraisalEngine,
    SyncToAsyncEmbedder,
    SyncToAsyncStore,
    as_async,
)
from emotional_memory.async_engine import AsyncEmotionalMemory
from emotional_memory.categorize import EmotionLabel, categorize_affect, label_tag
from emotional_memory.decay import DecayConfig
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.interfaces import (
    AffectiveStateStore,
    Embedder,
    MemoryStore,
    SequentialEmbedder,
)
from emotional_memory.interfaces_async import AsyncAppraisalEngine, AsyncEmbedder, AsyncMemoryStore
from emotional_memory.logging_config import configure_logging
from emotional_memory.models import EmotionalTag, Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodDecayConfig, MoodField
from emotional_memory.query_classifier import (
    LOCOMO_ROUTING,
    QUERY_TYPES,
    HeuristicQueryClassifier,
    LLMQueryClassifier,
    QueryClassifier,
)
from emotional_memory.resonance import ResonanceConfig, hebbian_strengthen, spreading_activation
from emotional_memory.retrieval import (
    AdaptiveWeightsConfig,
    QueryClassifierConfig,
    RetrievalBreakdown,
    RetrievalConfig,
    RetrievalExplanation,
    RetrievalSignals,
    compute_ape,
    update_prediction,
)
from emotional_memory.state import AffectiveState
from emotional_memory.state_stores.in_memory import InMemoryAffectiveStateStore
from emotional_memory.stores.in_memory import InMemoryStore

# Optional extras — name → (extra_name, module_path, attr)
# Loaded at import time when available; __getattr__ provides a clear error otherwise.
# PEP 562 (module __getattr__/__dir__) allows REPL/IDE discovery of all names.
_OPTIONAL_EXPORTS: dict[str, tuple[str, str, str]] = {
    "SQLiteStore": ("sqlite", "emotional_memory.stores.sqlite", "SQLiteStore"),
    "QdrantStore": ("qdrant", "emotional_memory.stores.qdrant", "QdrantStore"),
    "ChromaStore": ("chroma", "emotional_memory.stores.chroma", "ChromaStore"),
    "SQLiteAffectiveStateStore": (
        "sqlite",
        "emotional_memory.state_stores.sqlite",
        "SQLiteAffectiveStateStore",
    ),
    "RedisAffectiveStateStore": (
        "redis",
        "emotional_memory.state_stores.redis",
        "RedisAffectiveStateStore",
    ),
    "SentenceTransformerEmbedder": (
        "sentence-transformers",
        "emotional_memory.embedders.sentence_transformers",
        "SentenceTransformerEmbedder",
    ),
    "EmotionalMemoryMem0Backend": (
        "mem0",
        "emotional_memory.integrations.mem0",
        "EmotionalMemoryMem0Backend",
    ),
    "messages_to_content": (
        "mem0",
        "emotional_memory.integrations.mem0",
        "messages_to_content",
    ),
    "EmotionalMemoryChatHistory": (
        "langchain",
        "emotional_memory.integrations.langchain",
        "EmotionalMemoryChatHistory",
    ),
    "recommended_conversation_policy": (
        "langchain",
        "emotional_memory.integrations.langchain",
        "recommended_conversation_policy",
    ),
    "store_all_messages": (
        "langchain",
        "emotional_memory.integrations.langchain",
        "store_all_messages",
    ),
}

# Some internal modules are always importable (they lazy-import the external dep);
# for those, probe the external package directly to decide availability.
_EXTRA_PROBE: dict[str, str] = {"redis": "redis"}

for _opt_name, (_opt_extra, _opt_module, _opt_attr) in _OPTIONAL_EXPORTS.items():
    try:
        if _opt_probe := _EXTRA_PROBE.get(_opt_extra):
            importlib.import_module(_opt_probe)
        globals()[_opt_name] = getattr(importlib.import_module(_opt_module), _opt_attr)
    except ImportError:
        pass

if TYPE_CHECKING:
    # Static re-imports give type checkers (mypy/pyright) visibility of the
    # optional, extra-gated exports, so `from emotional_memory import SQLiteStore`
    # type-checks. At runtime these names are provided lazily by __getattr__ and
    # the eager-load loop above; names whose extra is absent are filtered out of
    # __all__ below. The redundant `as` aliases mark explicit re-exports (PEP 484).
    from emotional_memory.embedders.sentence_transformers import (
        SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    )
    from emotional_memory.integrations.langchain import (
        EmotionalMemoryChatHistory as EmotionalMemoryChatHistory,
    )
    from emotional_memory.integrations.langchain import (
        recommended_conversation_policy as recommended_conversation_policy,
    )
    from emotional_memory.integrations.langchain import (
        store_all_messages as store_all_messages,
    )
    from emotional_memory.integrations.mem0 import (
        EmotionalMemoryMem0Backend as EmotionalMemoryMem0Backend,
    )
    from emotional_memory.integrations.mem0 import (
        messages_to_content as messages_to_content,
    )
    from emotional_memory.state_stores.redis import (
        RedisAffectiveStateStore as RedisAffectiveStateStore,
    )
    from emotional_memory.state_stores.sqlite import (
        SQLiteAffectiveStateStore as SQLiteAffectiveStateStore,
    )
    from emotional_memory.stores.chroma import ChromaStore as ChromaStore
    from emotional_memory.stores.qdrant import QdrantStore as QdrantStore
    from emotional_memory.stores.sqlite import SQLiteStore as SQLiteStore

# Declared public API: always-available core names plus the optional, extra-gated
# names (the keys of _OPTIONAL_EXPORTS). A static literal is required for type
# checkers to resolve the export surface (reportUnsupportedDunderAll rejects
# computed __all__ values); unavailable optional names are filtered out at runtime.
__all__ = [
    "DIRECT_VAD_SCHEMA",
    "LOCOMO_ROUTING",
    "QUERY_TYPES",
    "SCHERER_CPM_SCHEMA",
    "AdaptiveWeightsConfig",
    "AffectiveMomentum",
    "AffectiveState",
    "AffectiveStateStore",
    "AppraisalDimension",
    "AppraisalEngine",
    "AppraisalSchema",
    "AppraisalVector",
    "AsyncAppraisalEngine",
    "AsyncEmbedder",
    "AsyncEmotionalMemory",
    "AsyncMemoryStore",
    "ChromaStore",
    "CoreAffect",
    "DecayConfig",
    "Embedder",
    "EmotionLabel",
    "EmotionalMemory",
    "EmotionalMemoryChatHistory",
    "EmotionalMemoryConfig",
    "EmotionalMemoryMem0Backend",
    "EmotionalTag",
    "GenericAppraisalVector",
    "HeuristicQueryClassifier",
    "InMemoryAffectiveStateStore",
    "InMemoryStore",
    "KeywordAppraisalEngine",
    "KeywordRule",
    "LLMAppraisalConfig",
    "LLMAppraisalEngine",
    "LLMCallable",
    "LLMQueryClassifier",
    "Memory",
    "MemoryStore",
    "MoodDecayConfig",
    "MoodField",
    "QdrantStore",
    "QueryClassifier",
    "QueryClassifierConfig",
    "RedisAffectiveStateStore",
    "ResonanceConfig",
    "ResonanceLink",
    "RetrievalBreakdown",
    "RetrievalConfig",
    "RetrievalExplanation",
    "RetrievalSignals",
    "SQLiteAffectiveStateStore",
    "SQLiteStore",
    "SentenceTransformerEmbedder",
    "SequentialEmbedder",
    "StaticAppraisalEngine",
    "SyncToAsyncAppraisalEngine",
    "SyncToAsyncEmbedder",
    "SyncToAsyncStore",
    "__version__",
    "as_async",
    "categorize_affect",
    "compute_ape",
    "configure_logging",
    "consolidation_strength",
    "hebbian_strengthen",
    "label_tag",
    "make_emotional_tag",
    "messages_to_content",
    "recommended_conversation_policy",
    "spreading_activation",
    "store_all_messages",
    "update_prediction",
]

if not TYPE_CHECKING:
    # Runtime: drop optional names whose extra did not resolve at import, so
    # `from emotional_memory import *` never raises and every advertised name is
    # importable. Type checkers skip this branch (TYPE_CHECKING is True for them).
    __all__ = [name for name in __all__ if name not in _OPTIONAL_EXPORTS or name in globals()]


def __getattr__(name: str) -> Any:
    """PEP 562: raise ImportError with install hint for unavailable optional extras."""
    if name in _OPTIONAL_EXPORTS:
        extra, _, _ = _OPTIONAL_EXPORTS[name]
        raise ImportError(
            f"{name!r} requires the '{extra}' extra. "
            f"Install with: pip install 'emotional_memory[{extra}]'"
        )
    raise AttributeError(f"module 'emotional_memory' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_OPTIONAL_EXPORTS))
