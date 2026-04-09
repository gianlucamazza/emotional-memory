"""Emotional memory library for LLMs based on Affective Field Theory."""

from importlib.metadata import version

__version__ = version("emotional_memory")

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.appraisal import (
    AppraisalEngine,
    AppraisalVector,
    StaticAppraisalEngine,
    consolidation_strength,
)
from emotional_memory.decay import DecayConfig
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.interfaces import Embedder, MemoryStore
from emotional_memory.models import EmotionalTag, Memory, ResonanceLink, make_emotional_tag
from emotional_memory.resonance import ResonanceConfig
from emotional_memory.retrieval import RetrievalConfig
from emotional_memory.state import AffectiveState
from emotional_memory.stimmung import StimmungField
from emotional_memory.stores.in_memory import InMemoryStore

__all__ = [
    "AffectiveMomentum",
    "AffectiveState",
    "AppraisalEngine",
    "AppraisalVector",
    "CoreAffect",
    "DecayConfig",
    "Embedder",
    "EmotionalMemory",
    "EmotionalMemoryConfig",
    "EmotionalTag",
    "InMemoryStore",
    "Memory",
    "MemoryStore",
    "ResonanceConfig",
    "ResonanceLink",
    "RetrievalConfig",
    "StaticAppraisalEngine",
    "StimmungField",
    "__version__",
    "consolidation_strength",
    "make_emotional_tag",
]
