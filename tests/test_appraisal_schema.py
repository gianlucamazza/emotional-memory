"""Tests for AppraisalDimension, AppraisalSchema, and SCHERER_CPM_SCHEMA."""

import pytest
from pydantic import ValidationError

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal_schema import (
    SCHERER_CPM_SCHEMA,
    AppraisalDimension,
    AppraisalSchema,
)


class TestAppraisalDimension:
    def test_valid_construction(self):
        dim = AppraisalDimension(
            name="valence", range=(-1.0, 1.0), neutral=0.0, description="test"
        )
        assert dim.name == "valence"
        assert dim.range == (-1.0, 1.0)
        assert dim.neutral == 0.0

    def test_range_must_be_ordered(self):
        with pytest.raises(ValidationError):
            AppraisalDimension(name="bad", range=(1.0, -1.0), neutral=0.0, description="bad")

    def test_range_equal_rejected(self):
        with pytest.raises(ValidationError):
            AppraisalDimension(name="bad", range=(0.0, 0.0), neutral=0.0, description="bad")

    def test_frozen(self):
        dim = AppraisalDimension(name="x", range=(0.0, 1.0), neutral=0.5, description="d")
        with pytest.raises(ValidationError):
            dim.name = "y"  # type: ignore[misc]


class TestAppraisalSchemaJsonSchema:
    def _make_simple_schema(self) -> AppraisalSchema:
        dims = (
            AppraisalDimension(
                name="desirability", range=(-1.0, 1.0), neutral=0.0, description="D"
            ),
            AppraisalDimension(
                name="praiseworthiness", range=(-1.0, 1.0), neutral=0.0, description="P"
            ),
            AppraisalDimension(
                name="appealingness", range=(0.0, 1.0), neutral=0.5, description="A"
            ),
        )
        return AppraisalSchema(
            name="occ_subset",
            dimensions=dims,
            system_prompt="Rate the event. Return JSON.",
            project_to_core_affect=lambda d: CoreAffect(
                valence=0.6 * d["desirability"] + 0.4 * d["praiseworthiness"],
                arousal=0.5 * abs(d["desirability"]),
                dominance=0.5,
            ),
        )

    def test_json_schema_shape(self):
        schema = self._make_simple_schema()
        js = schema.to_json_schema()
        assert js["type"] == "object"
        assert "properties" in js
        assert "required" in js
        assert js["additionalProperties"] is False

    def test_all_dims_in_required(self):
        schema = self._make_simple_schema()
        js = schema.to_json_schema()
        assert js["required"] == ["desirability", "praiseworthiness", "appealingness"]

    def test_dim_ranges_in_properties(self):
        schema = self._make_simple_schema()
        js = schema.to_json_schema()
        d = js["properties"]["desirability"]
        assert d["minimum"] == -1.0
        assert d["maximum"] == 1.0
        a = js["properties"]["appealingness"]
        assert a["minimum"] == 0.0
        assert a["maximum"] == 1.0

    def test_project_to_core_affect_returns_core_affect(self):
        schema = self._make_simple_schema()
        result = schema.project_to_core_affect(
            {"desirability": 0.8, "praiseworthiness": 0.6, "appealingness": 0.5}
        )
        assert isinstance(result, CoreAffect)
        assert -1.0 <= result.valence <= 1.0
        assert 0.0 <= result.arousal <= 1.0

    def test_repr(self):
        schema = self._make_simple_schema()
        r = repr(schema)
        assert "occ_subset" in r
        assert "desirability" in r


class TestSchereCpmSchema:
    def test_has_five_dimensions(self):
        assert len(SCHERER_CPM_SCHEMA.dimensions) == 5

    def test_dimension_names(self):
        names = [d.name for d in SCHERER_CPM_SCHEMA.dimensions]
        assert "novelty" in names
        assert "goal_relevance" in names
        assert "coping_potential" in names
        assert "norm_congruence" in names
        assert "self_relevance" in names

    def test_schema_name(self):
        assert SCHERER_CPM_SCHEMA.name == "scherer_cpm"

    def test_system_prompt_not_empty(self):
        assert len(SCHERER_CPM_SCHEMA.system_prompt) > 50

    def test_json_schema_has_all_required_fields(self):
        js = SCHERER_CPM_SCHEMA.to_json_schema()
        required = js["required"]
        for name in (
            "novelty",
            "goal_relevance",
            "coping_potential",
            "norm_congruence",
            "self_relevance",
        ):
            assert name in required

    def test_project_neutral_gives_near_zero_valence(self):
        neutral_dims = {
            "novelty": 0.0,
            "goal_relevance": 0.0,
            "coping_potential": 0.5,
            "norm_congruence": 0.0,
            "self_relevance": 0.0,
        }
        ca = SCHERER_CPM_SCHEMA.project_to_core_affect(neutral_dims)
        assert isinstance(ca, CoreAffect)
        assert abs(ca.valence) < 0.05

    def test_project_positive_dims_gives_positive_valence(self):
        positive_dims = {
            "novelty": 0.5,
            "goal_relevance": 1.0,
            "coping_potential": 1.0,
            "norm_congruence": 1.0,
            "self_relevance": 1.0,
        }
        ca = SCHERER_CPM_SCHEMA.project_to_core_affect(positive_dims)
        assert ca.valence > 0.5
