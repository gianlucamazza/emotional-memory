import pytest
from pydantic import ValidationError

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.appraisal import AppraisalVector
from emotional_memory.models import EmotionalTag, Memory, ResonanceLink, make_emotional_tag
from emotional_memory.stimmung import StimmungField


def _tag() -> EmotionalTag:
    return make_emotional_tag(
        core_affect=CoreAffect(valence=0.3, arousal=0.5),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=0.7,
    )


class TestResonanceLink:
    def test_valid_link_types(self):
        for lt in ("semantic", "emotional", "temporal", "causal", "contrastive"):
            link = ResonanceLink(source_id="a", target_id="b", strength=0.5, link_type=lt)
            assert link.link_type == lt

    def test_invalid_link_type(self):
        with pytest.raises(ValidationError):
            ResonanceLink(source_id="a", target_id="b", strength=0.5, link_type="invalid")

    def test_frozen(self):
        link = ResonanceLink(source_id="a", target_id="b", strength=0.5, link_type="semantic")
        with pytest.raises(ValidationError):
            link.strength = 0.9  # type: ignore[misc]


class TestEmotionalTag:
    def test_defaults(self):
        tag = _tag()
        assert tag.retrieval_count == 0
        assert tag.reconsolidation_count == 0
        assert tag.last_retrieved is None
        assert tag.appraisal is None
        assert tag.resonance_links == []

    def test_consolidation_strength_clamped(self):
        tag = make_emotional_tag(
            core_affect=CoreAffect.neutral(),
            momentum=AffectiveMomentum.zero(),
            stimmung=StimmungField.neutral(),
            consolidation_strength=1.5,
        )
        assert tag.consolidation_strength == 1.0

    def test_with_appraisal(self):
        appraisal = AppraisalVector.neutral()
        tag = make_emotional_tag(
            core_affect=CoreAffect.neutral(),
            momentum=AffectiveMomentum.zero(),
            stimmung=StimmungField.neutral(),
            consolidation_strength=0.5,
            appraisal=appraisal,
        )
        assert tag.appraisal == appraisal

    def test_serialization_roundtrip(self):
        tag = _tag()
        data = tag.model_dump()
        restored = EmotionalTag.model_validate(data)
        assert restored.core_affect.valence == tag.core_affect.valence
        assert restored.consolidation_strength == tag.consolidation_strength
        assert restored.retrieval_count == tag.retrieval_count


class TestMemory:
    def test_auto_id(self):
        m1 = Memory.create(content="hello", tag=_tag())
        m2 = Memory.create(content="world", tag=_tag())
        assert m1.id != m2.id

    def test_defaults(self):
        m = Memory.create(content="test", tag=_tag())
        assert m.embedding is None
        assert m.metadata == {}

    def test_with_embedding(self):
        emb = [0.1, 0.2, 0.3]
        m = Memory.create(content="test", tag=_tag(), embedding=emb)
        assert m.embedding == emb

    def test_serialization_roundtrip(self):
        m = Memory.create(content="round trip test", tag=_tag(), metadata={"key": "val"})
        data = m.model_dump()
        restored = Memory.model_validate(data)
        assert restored.id == m.id
        assert restored.content == m.content
        assert restored.metadata == m.metadata
        assert restored.tag.core_affect.valence == m.tag.core_affect.valence
