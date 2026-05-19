"""Emotional memory library for LLMs based on Affective Field Theory."""

import contextlib
from importlib.metadata import version

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
from emotional_memory.state_stores.redis import RedisAffectiveStateStore
from emotional_memory.stores.in_memory import InMemoryStore

_sqlite_available = False
with contextlib.suppress(ImportError):
    from emotional_memory.stores.sqlite import SQLiteStore as SQLiteStore

    _sqlite_available = True

_qdrant_available = False
with contextlib.suppress(ImportError):
    from emotional_memory.stores.qdrant import QdrantStore as QdrantStore

    _qdrant_available = True

_chroma_available = False
with contextlib.suppress(ImportError):
    from emotional_memory.stores.chroma import ChromaStore as ChromaStore

    _chroma_available = True

_sqlite_state_available = False
with contextlib.suppress(ImportError):
    from emotional_memory.state_stores.sqlite import (
        SQLiteAffectiveStateStore as SQLiteAffectiveStateStore,
    )

    _sqlite_state_available = True

_sentence_transformers_available = False
with contextlib.suppress(ImportError):
    from emotional_memory.embedders.sentence_transformers import (
        SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    )

    _sentence_transformers_available = True

with contextlib.suppress(ImportError):
    from emotional_memory.integrations.mem0 import (
        EmotionalMemoryMem0Backend as EmotionalMemoryMem0Backend,
    )
    from emotional_memory.integrations.mem0 import (
        messages_to_content as messages_to_content,
    )

_langchain_available = False
with contextlib.suppress(ImportError):
    from emotional_memory.integrations.langchain import (
        EmotionalMemoryChatHistory,
        recommended_conversation_policy,
        store_all_messages,
    )

    _langchain_available = True

__all__ = [
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
